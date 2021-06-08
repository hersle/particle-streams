using LinearAlgebra
using Printf
using ProgressMeter
using Profile
using GLMakie
using StaticArrays
GLMakie.AbstractPlotting.inline!(false) # show window while animating

struct Rectangle
	x1::Float64
	y1::Float64
	x2::Float64
	y2::Float64
end

width(rect::Rectangle) = rect.x2 - rect.x1

struct Wall
	p1::Tuple{Float64, Float64}
	p2::Tuple{Float64, Float64}
	n::Tuple{Float64, Float64}

	function Wall(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64})
		n = (-(p2[2]-p1[2]), p2[1]-p1[1]) # normal is the vector that points normally left out of (p1 - p2)
		n = n ./ norm(n)
		return new(p1, p2, n)
	end
end

@Base.kwdef struct Parameters{F1<:Function, F2<:Function, WN} # allow construction with Parameters(N=...)
	N::Int
	T::Float64

	bounds::Rectangle

	radius::Float64
	spawn_radius::Float64

	position_spawner::F1 # F1 makes compiler figure out types # https://stackoverflow.com/questions/52992375/julia-mutable-struct-with-attribute-which-is-a-function-and-code-warntype
	velocity_spawner::F2 # F2 makes compiler figure out types
	max_velocity::Float64 # promise thet velocity_spawner never spawns with larger velocity

	walls::SVector{WN, Wall} # use SVector to avoid allocations
end

struct Simulation
	params::Parameters

	times::AbstractArray
	positions::Array{Tuple{Float64, Float64}, 2}
	velocities::Array{Tuple{Float64, Float64}, 2}
	alive::Array{Bool, 2}

	trajectories::Array{Array{Tuple{Float64, Float64, Float64}}} # (time, x, y) for each particle life

	ncellsx::Int
	ncellsy::Int
end

function out_of_bounds(rect::Rectangle, pos::Tuple{Float64, Float64})
    x, y = pos[1], pos[2]
    return x < rect.x1 || x > rect.x2 || y < rect.y1 || y > rect.y2
end

function pos2cell(pos, rect::Rectangle, ncellsx, ncellsy) 
	cx = clamp(1 + (pos[1]-rect.x1)/(rect.x2-rect.x1)*ncellsx, 1, ncellsx)
	cy = clamp(1 + (pos[2]-rect.y1)/(rect.y2-rect.y1)*ncellsy, 1, ncellsy)
	return Int(floor(cx)), Int(floor(cy))
end

function pos2cells(pos, rect::Rectangle, ncellsx, ncellsy)
	# assumes that cells are chosen so one object can be at most in 4 cells
	cx1 = Int(floor(1 + (pos[1]-rect.x1)/(rect.x2-rect.x1)*ncellsx - 0.5)) # -0.5 makes this the "leftmost" cell
	cy1 = Int(floor(1 + (pos[2]-rect.y1)/(rect.y2-rect.y1)*ncellsy - 0.5))
	cx1 = clamp(cx1, 1, ncellsx-1)
	cy1 = clamp(cy1, 1, ncellsy-1)
	cx2 = cx1 + 1
	cy2 = cy1 + 1
	return cx1, cy1, cx2, cy2
end

function scatter_particles(pos1, vel1, pos2, vel2, radius)
	r = norm(pos2 .- pos1)
	if r < 2*radius && dot(vel2.-vel1,pos2.-pos1) <= 0
		dvel1 = dot(vel1.-vel2,pos1.-pos2) / (r*r) .* (pos1 .- pos2)
		#println("$(norm(vel1)^2 + norm(vel2)^2 - norm(vel1 .- dvel1)^2 - norm(vel2 .+ dvel1)^2)") # v1^2 + v2^2 is conserved (verify by uncommenting)
		vel1, vel2 = vel1 .- dvel1, vel2 .+ dvel1
		return vel1, vel2, true
	else
		return vel1, vel2, false
	end
end

function closest_point_on_line(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64}, p::Tuple{Float64, Float64})
	return p1 .+ (p2 .- p1) .* (dot(p .- p1, p2 .- p1) / dot(p2 .- p1, p2 .- p1))
end

function signed_distance_from_line(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64}, n::Tuple{Float64, Float64}, p::Tuple{Float64, Float64})
	return dot(p .- closest_point_on_line(p1, p2, p), n)
end

function scatter(pos::Tuple{Float64, Float64}, vel::Tuple{Float64, Float64}, wall::Wall, radius::Float64)
	dot1 = dot(pos .- wall.p1, wall.p2 .- wall.p1) # check if inside wall endpoints
	dot2 = dot(pos .- wall.p2, wall.p1 .- wall.p2)
	sdist = signed_distance_from_line(wall.p1, wall.p2, wall.n, pos)
	wallvel = dot(vel, wall.n) # velocity towards wall
	if dot1 >= 0 && dot2 >= 0 && sdist <= radius && wallvel <= 0
		return vel .- wall.n .* (2*wallvel), true
	else
		return vel, false
	end
end

function scatter_wall(pos, vel, radius)
	if pos[2] < 0 && vel[2] < 0
		return (vel[1], -vel[2]), true # reflect y-velocity
	else
		return vel, false
	end
end

function simulate(params::Parameters; sample=false, write_trajectories=false, animation_path="", frameskip=1, anim_t1=0, anim_t2=params.T, grid=false)
    dt = 0.5 * params.radius / params.max_velocity # 0.8 safety factor # 1.0 would mean particle centers could overlap in one step
    times = 0:dt:params.T
    NT = length(times)
    
	outsidepos = (params.bounds.x1 - 10 * params.radius, params.bounds.y1 - 10 * params.radius)
    positions = fill(outsidepos, params.N)
    velocities = fill((0.0, 0.0), params.N)
	alive = fill(false, params.N)
    
	sample_size = sample ? (params.N, NT) : (0, 0)
	positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, sample_size)
	velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, sample_size)
	alive_samples = Array{Bool, 2}(undef, sample_size)

	has_scattered = Array{Bool}(undef, params.N) # whether a particle has scattered the last time step

	# aim for 1 particle in each cell, so one particle can be in at most 4 cells
	max_radius = max(params.radius, params.spawn_radius)
	min_radius = min(params.radius, params.spawn_radius)
	ncellsx = Int(floor((params.bounds.x2-params.bounds.x1) / (2*max_radius))-1) # floor & reduce, so at most 4 particles in each cell
	ncellsy = Int(floor((params.bounds.y2-params.bounds.y1) / (2*max_radius))-1)
	cellwidth = (params.bounds.x2-params.bounds.x1) / ncellsx
	cellheight = (params.bounds.y2-params.bounds.y1) / ncellsy

	max_parts_per_cell = 16 * Int(round(9*cellwidth*cellheight / (pi*params.radius^2))) # assume complete filling for simple upper bound

	# TODO: use some form of map type?
	# maps from cell -> particle and particle -> cell
	celllen = zeros(Int, (ncellsx, ncellsy))
	cell2part = Array{Tuple{Int, Int}, 3}(undef, ncellsx, ncellsy, max_parts_per_cell) # (cx, cy, ci) -> (particle id, cell #1-4)
	part2cell = Array{Tuple{Int, Int, Int}, 2}(undef, params.N, 4) # (particle id, cell #1-4) -> (cx, cy, ci)

	animation_path_prefix = "/home/herman/Videos/particle_collisions/"
	animation_path = animation_path_prefix * animation_path
	println("Write trajectories:  $write_trajectories")
	println("Write animation:     $animation_path")
	println("Number of cells:     ($ncellsx, $ncellsy)")
	println("Max parts. per cell: $max_parts_per_cell")

	function addpart(n)
		pos = positions[n]
		cx1, cy1, cx2, cy2 = pos2cells(pos, params.bounds, ncellsx, ncellsy)
		# println("$cx1 $cy1 $cx2 $cy2")
		celllen[cx1,cy1] += 1
		celllen[cx1,cy2] += 1
		celllen[cx2,cy1] += 1
		celllen[cx2,cy2] += 1
		cell2part[cx1,cy1,celllen[cx1,cy1]] = (n, 1) # add to all four cells
		cell2part[cx1,cy2,celllen[cx1,cy2]] = (n, 2)
		cell2part[cx2,cy1,celllen[cx2,cy1]] = (n, 3)
		cell2part[cx2,cy2,celllen[cx2,cy2]] = (n, 4)
		part2cell[n,1] = (cx1,cy1,celllen[cx1,cy1]) # remember which cells it is in
		part2cell[n,2] = (cx1,cy2,celllen[cx1,cy2])
		part2cell[n,3] = (cx2,cy1,celllen[cx2,cy1])
		part2cell[n,4] = (cx2,cy2,celllen[cx2,cy2])
	end

	function rempart(n)
		for i in 1:4
			cx, cy, ci = part2cell[n,i] # this cell contains particle n
			cell2part[cx,cy,ci] = cell2part[cx,cy,celllen[cx,cy]] # remove (n,i) from cell
			part2cell[cell2part[cx,cy,ci][1],cell2part[cx,cy,ci][2]] = (cx,cy,ci) # update reverse map on new particle at (cx,cy,ci)
			celllen[cx,cy] -= 1
		end
	end

	function kill_particle(n, iter)
		rempart(n)
		alive[n] = false
		if write_trajectories
			log_trajectory(n, iter)
		end
		positions[n] = outsidepos # move outside area (just to remove from plot)
	end

	part2id = Array{Int}(undef, params.N)
	trajectories = Array{Array{Tuple{Float64, Float64, Float64}}}(undef, 0) # (time, x, y) for each particle life

	# when spawning
	function start_trajectory(n)
		push!(trajectories, Array{Tuple{Float64, Float64, Float64}}(undef, 0))
		part2id[n] = length(trajectories)
	end

	# when living
	function log_trajectory(n, iter)
		push!(trajectories[part2id[n]], (times[iter], positions[n][1], positions[n][2]))
	end

	function spawn_particle(n, iter, pos, vel)
		alive[n] = true
		positions[n] = pos
		velocities[n] = vel
		addpart(n) # add to cells
		if write_trajectories
			start_trajectory(n)
			log_trajectory(n, iter)
		end
	end

	function position_is_available(pos1)
		cx1, cy1, cx2, cy2 = pos2cells(pos1, params.bounds, ncellsx, ncellsy)
		for cx in cx1:cx2 # TODO: create some form of cleaner iteration
			for cy in cy1:cy2
				for i in 1:celllen[cx,cy]
					n2 = cell2part[cx,cy,i][1]
					pos2 = positions[n2]
					if norm(pos2 .- pos1) < 2 * params.spawn_radius
						return false
					end
				end
			end
		end
		return true
	end
    
	function step(iter::Int)
		# kill dead particles and (try to) respawn them
		print("\rSimulating time step $iter / $NT ...")

		fill!(has_scattered, false)

		for n1 in 1:params.N
			if alive[n1] && out_of_bounds(params.bounds, positions[n1])
				kill_particle(n1, iter)
			end
			if !alive[n1]
				pos = params.position_spawner(params, n1, times[iter])
				if position_is_available(pos)
					vel = params.velocity_spawner(params, n1, times[iter], pos)
					@assert dot(vel, vel) <= params.max_velocity^2+1e-10 "Spawned particle with speed $(norm(vel)) > $(params.max_velocity) = max_velocity"
					spawn_particle(n1, iter, pos, vel)
				end
			end

			# sample current positions and velocities
			if sample
				positions_samples[n1,iter] = positions[n1]
				velocities_samples[n1,iter] = velocities[n1]
				alive_samples[n1,iter] = alive[n1]
			end

			if alive[n1]
				# integrate particle positions (and update cell locations)
				lastpos = positions[n1]
				newpos = positions[n1] .+ velocities[n1] .* dt
				positions[n1] = newpos
				rempart(n1) # remove from current cell (based on prev pos)
				addpart(n1) # add to next cell (based on current pos)

				# particle - particle interactions
				pos1, vel1 = positions[n1], velocities[n1]
				cx1, cy1, cx2, cy2 = pos2cells(pos1, params.bounds, ncellsx, ncellsy)
				for cx in cx1:cx2 # TODO: create some form of cleaner iteration
					for cy in cy1:cy2
						for i in 1:celllen[cx,cy]
							n2 = cell2part[cx,cy,i][1]
							pos2, vel2 = positions[n2], velocities[n2]
							if alive[n2] && n2 > n1
								vel1, vel2, scattered = scatter_particles(pos1, vel1, pos2, vel2, params.radius) # velocities[n1], velocities[n2] = scatter_particles() causes errors!
								velocities[n1], velocities[n2] = vel1, vel2
								has_scattered[n1] = has_scattered[n1] || scattered
							end
						end
					end
				end

				# particle - wall interactions
				for wall in params.walls
					vel1, scattered = scatter(pos1, vel1, wall, params.radius)
					velocities[n1] = vel1
					has_scattered[n1] = has_scattered[n1] || scattered
				end

				if has_scattered[n1] && write_trajectories
					log_trajectory(n1, iter) # log trajectory only when scattering
				end
			end
		end
	end

	figure = Figure(resolution=(1000*(params.bounds.x2-params.bounds.x1)/(params.bounds.y2-params.bounds.y1), 1000))
	axis = Axis(figure[1,1], 
		xlabel="W = $(params.bounds.x2-params.bounds.x1)", xminorticks=IntervalsBetween(ncellsx), xminorgridvisible=true,
		ylabel="H = $(params.bounds.y2-params.bounds.y1)", yminorticks=IntervalsBetween(ncellsy), yminorgridvisible=true,
	)
	axis.xticks = [params.bounds.x1, params.bounds.x2]
	axis.yticks = [params.bounds.y1, params.bounds.y2]
	hidexdecorations!(axis, label=false, minorgrid=!grid)
	hideydecorations!(axis, label=false, minorgrid=!grid)
	xlims!(axis, params.bounds.x1, params.bounds.x2)
	ylims!(axis, params.bounds.y1, params.bounds.y2)

	animation_positions = Node(positions)
	scatter!(axis, animation_positions, markersize=2*params.radius, markerspace=AbstractPlotting.SceneSpace, color=:black)

	# draw static geometry
	wallpoints = Array{Tuple{Float64, Float64}}(undef, 0)
	for wall in params.walls
		push!(wallpoints, wall.p1)
		push!(wallpoints, wall.p2)
	end
	linesegments!(axis, wallpoints)

	# animation stuff
	# find closest indices in times-array corresponding to t1 and t2
	f1 = findmin(abs.(times .- anim_t1))[2]
	f2 = findmin(abs.(times .- anim_t2))[2]
	frames = f1:frameskip:f2
	fps = Int(round(length(frames) / (anim_t2 - anim_t1))) # make duration equal to simulation time in seconds

	display(figure)

	function animation_step(iter::Int, io)
		step(iter)
		if iter in frames
			animation_positions[] = positions
			nalive = sum(alive)
			t = round(times[iter], digits=1)
			R = round(params.radius, digits=4)
			S = round(params.spawn_radius, digits=4)
			axis.title = "N = $nalive   t = $t   R = $R   S = $S"
			if io != nothing
				recordframe!(io)
			end
		end
	end

	if animation_path == ""
		for iter in 1:NT
			animation_step(iter, nothing)
		end
	else
		dirname, filename = Base.Filesystem.splitdir(animation_path)
		Base.Filesystem.mkpath(dirname)
		record(figure, animation_path, framerate=fps) do io
			for iter in 1:NT
				animation_step(iter, io)
			end
		end
	end

	println() # finish progress printing
    return Simulation(params, times, positions_samples, velocities_samples, alive_samples, trajectories, ncellsx, ncellsy)
end

#= TODO: needs update with newest changes
function plot_trajectories(sim, which)
	plot()
	setmarkersize(1.0)
	for trajectory in sim.trajectories[which]
		x = [txy[2] for txy in trajectory]
		y = [txy[3] for txy in trajectory]
		oplot(x, y, "-*o")
	end
	oplot(
		size=(800*sim.params.width/sim.params.height, 800),
		xlim=(-sim.params.width/2, +sim.params.width/2), ylim=(0, sim.params.height),
	)
end
=#

function write_trajectories(sim, path)
	f = open(path, "w")

	# write explanation comment
	write(f, "# n t1 x1 y1 t2 x2 y2 t3 x3 y3 ...\n")

	for (n, trajectory) in enumerate(sim.trajectories)
		write(f, "$n")
		for (t, x, y) in trajectory
			write(f, " $t $x $y")
		end
		write(f, "\n")
	end

	close(f)
end

# TODO: is it randomized spawning position or velocity that causes symmetry breaking?

function crosswalls(bounds::Rectangle, wx::Float64, wy::Float64=wx)
	x1, x4 = bounds.x1, bounds.x2
	x2, x3 = (x1+x4)/2 - wx/2, (x1+x4)/2 + wx/2
	y1, y4 = bounds.y1, bounds.y2
	y2, y3 = (y1+y4)/2 - wy/2, (y1+y4)/2 + wy/2
	nw1, nw2, nw3 = (x2, y4), (x2, y3), (x1, y3)
	ne1, ne2, ne3 = (x4, y3), (x3, y3), (x3, y4)
	se1, se2, se3 = (x3, y1), (x3, y2), (x4, y2)
	sw1, sw2, sw3 = (x1, y2), (x2, y2), (x2, y1)
	return SVector(
		Wall(nw1, nw2), Wall(nw2, nw3),
		Wall(ne1, ne2), Wall(ne2, ne3),
		Wall(sw1, sw2), Wall(sw2, sw3),
		Wall(se1, se2), Wall(se2, se3),
	)
end

function main()
	radius = 0.5
	crosswidth = 25.0
	bounds = Rectangle(-50, -50, +50, +50)
	params = Parameters(
		N = 40000,
		T = 100.0,
		bounds = bounds,
		radius = radius,
		spawn_radius = 1.0 * radius,
		position_spawner = (p::Parameters, n::Int, t::Float64) -> (isodd(n) ? bounds.x1 : bounds.x2, (-crosswidth/2+radius)+(crosswidth-2*radius)*rand()),
		velocity_spawner = (p::Parameters, n::Int, t::Float64, pos::Tuple{Float64, Float64}) -> (y = pos[2]; ang=-pi/6+rand()*2*pi/6 + pi*iseven(n); (3.5 + 1.0*rand()) .* (cos(ang), sin(ang))),
		max_velocity = 4.5,
		walls = crosswalls(bounds, crosswidth),
		#walls = SVector(Wall((bounds.x1, bounds.y1), (bounds.x2, bounds.y1)))
	)
	simulate(params, animation_path="animation.mkv", frameskip=4)
end
main()
