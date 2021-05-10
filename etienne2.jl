using LinearAlgebra
using Printf
using ProgressMeter
using GLMakie
using Profile
GLMakie.AbstractPlotting.inline!(true) # do not show window while animating

@Base.kwdef struct Parameters # allow construction with Parameters(N=...)
	N::Int
	T::Float64

	width::Float64
	height::Float64

	radius::Float64
	spawn_radius::Float64

	position_spawner::Function
	velocity_spawner::Function
	max_velocity::Float64 # promise thet velocity_spawner never spawns with larger velocity
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

function out_of_bounds(width, height, pos)
    x, y = pos[1], pos[2]
    return x < -width/2 || x > +width/2 || y > +height || y < -50
end

function pos2cell(pos, width, height, ncellsx, ncellsy) 
	cx = clamp(1 + (pos[1]+width/2)/width*ncellsx, 1, ncellsx)
	cy = clamp(1 + pos[2]/height*ncellsy, 1, ncellsy)
	return Int(floor(cx)), Int(floor(cy))
end

function pos2cells(pos, width, height, ncellsx, ncellsy)
	# assumes that cells are chosen so one object can be at most in 4 cells
	cx1 = Int(floor(1 + (pos[1]+width/2)/width*ncellsx - 0.5)) # -0.5 makes this the "leftmost" cell
	cy1 = Int(floor(1 + pos[2]/height*ncellsy - 0.5))
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
		vel1, vel2 = vel1 .- dvel1, vel2 .+ dvel1
		return vel1, vel2, true
	else
		return vel1, vel2, false
	end
end

function scatter_wall(pos, vel, radius)
	if pos[2] < 0 && vel[2] < 0
		return (vel[1], -vel[2]), true # reflect y-velocity
	else
		return vel, false
	end
end

function simulate(params::Parameters; sample=false, write_trajectories=false, animation_path="anim.mkv")
    dt = 0.1 * params.radius / params.max_velocity # 0.7 safety factor # 1.0 would mean particle centers could overlap in one step
    times = 0:dt:params.T
    NT = length(times)
    
	outsidepos = (-params.width/2 - 10 * params.radius, - 10 * params.radius)
    positions = fill(outsidepos, params.N)
    velocities = fill((0.0, 0.0), params.N)
	alive = fill(false, params.N)
    
	sample_size = sample ? (params.N, NT) : (0, 0)
	positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, sample_size)
	velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, sample_size)
	alive_samples = Array{Bool, 2}(undef, sample_size)

	# aim for 1 particle in each cell, so one particle can be in at most 4 cells
	max_radius = max(params.radius, params.spawn_radius)
	min_radius = min(params.radius, params.spawn_radius)
	ncellsx = Int(floor(params.width / (2*max_radius))-1) # floor & reduce, so at most 4 particles in each cell
	ncellsy = Int(floor(params.height / (2*max_radius))-1)
	cellwidth = params.width / ncellsx
	cellheight = params.height / ncellsy

	max_parts_per_cell = 16 * Int(round(9*cellwidth*cellheight / (pi*params.radius^2))) # assume complete filling for simple upper bound

	# TODO: use some form of map type?
	# maps from cell -> particle and particle -> cell
	celllen = zeros(Int, (ncellsx, ncellsy))
	cell2part = Array{Tuple{Int, Int}, 3}(undef, ncellsx, ncellsy, max_parts_per_cell) # (cx, cy, ci) -> (particle id, cell #1-4)
	part2cell = Array{Tuple{Int, Int, Int}, 2}(undef, params.N, 4) # (particle id, cell #1-4) -> (cx, cy, ci)

	println("Write trajectories:  $write_trajectories")
	println("Number of cells:     ($ncellsx, $ncellsy)")
	println("Max parts. per cell: $max_parts_per_cell")

	function addpart(n)
		pos = positions[n]
		cx1, cy1, cx2, cy2 = pos2cells(pos, params.width, params.height, ncellsx, ncellsy)
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
		positions[n] = (-1000 * params.width/2, -1000) # move outside area (just to remove from plot)
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
		cx1, cy1, cx2, cy2 = pos2cells(pos1, params.width, params.height, ncellsx, ncellsy)
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

	# animation stuff
	figure = Figure(resolution=(600*params.width/params.height, 600))
	axis = Axis(figure[1,1], 
		xlabel="W = $(params.width)",  xminorticks=IntervalsBetween(ncellsx), xminorgridvisible=true,
		ylabel="H = $(params.height)", yminorticks=IntervalsBetween(ncellsy), yminorgridvisible=true,
	)
	axis.xticks = [-params.width/2, +params.width/2]
	axis.yticks = [0, +params.height]
	hidexdecorations!(axis, label=false, minorgrid=false)
	hideydecorations!(axis, label=false, minorgrid=false)
	xlims!(axis, -params.width/2, +params.width/2)
	ylims!(axis, 0, +params.height)

	animation_positions = Node(positions)
	scatter!(axis, animation_positions, markersize=2*params.radius, markerspace=AbstractPlotting.SceneSpace, color=:red)

	frameskip = 1
	frames = 1:frameskip:NT
	fps = Int(round(length(frames) / params.T)) # make duration equal to simulation time in seconds
    
	record(figure, animation_path, framerate=fps) do io
		for iter in 1:NT
			# kill dead particles and (try to) respawn them
			print("\rSimulating time step $iter / $NT ...")

			for n1 in 1:params.N
				if alive[n1] && out_of_bounds(params.width, params.height, positions[n1])
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
					newpos = positions[n1] .+ velocities[n1] .* dt
					positions[n1] = newpos
					rempart(n1) # remove from current cell (based on prev pos)
					addpart(n1) # add to next cell (based on current pos)

					has_scattered = false

					# particle - particle interactions
					pos1, vel1 = positions[n1], velocities[n1]
					cx1, cy1, cx2, cy2 = pos2cells(pos1, params.width, params.height, ncellsx, ncellsy)
					for cx in cx1:cx2 # TODO: create some form of cleaner iteration
						for cy in cy1:cy2
							for i in 1:celllen[cx,cy]
								n2 = cell2part[cx,cy,i][1]
								pos2, vel2 = positions[n2], velocities[n2]
								if alive[n2] && n2 > n1
									vel1, vel2, scattered = scatter_particles(pos1, vel1, pos2, vel2, params.radius) # velocities[n1], velocities[n2] = scatter_particles() causes errors!
									velocities[n1], velocities[n2] = vel1, vel2
									has_scattered = has_scattered || scattered
								end
							end
						end
					end

					# particle - wall interactions
					vel1, scattered = scatter_wall(pos1, vel1, params.radius)
					velocities[n1] = vel1
					has_scattered = has_scattered || scattered

					if has_scattered && write_trajectories
						log_trajectory(n1, iter) # log trajectory only when scattering
					end
				end
			end

			if iter in frames
				animation_positions[] = positions
				nalive = sum(alive)
				t = round(times[iter], digits=1)
				R = params.radius
				S = params.spawn_radius
				axis.title = "N = $nalive        t = $t        R = $R        S = $S"
				recordframe!(io)
			end
		end
	end
	return figure
	println() # finish progress printing
    
    return Simulation(params, times, positions_samples, velocities_samples, alive_samples, trajectories, ncellsx, ncellsy)
end

function animate_trajectories(sim::Simulation; t1=0, t2=sim.times[end], path="anim.mkv", frameskip=1)
	# TODO: force clear, new figure or something?

	# find closest indices in times-array corresponding to t1 and t2
	f1 = findmin(abs.(sim.times .- t1))[2]
	f2 = findmin(abs.(sim.times .- t2))[2]
	frames = f1:frameskip:f2
	nf = length(frames)

	figure = Figure(resolution=(600*sim.params.width/sim.params.height, 600))
	axis = Axis(figure[1,1], 
		xlabel="W = $(sim.params.width)",   xminorticks=IntervalsBetween(sim.ncellsx), xminorgridvisible=true,
		ylabel="H = $(sim.params.height)", yminorticks=IntervalsBetween(sim.ncellsy), yminorgridvisible=true,
	)
	axis.xticks = [-sim.params.width/2, +sim.params.width/2]
	axis.yticks = [0, +sim.params.height]
	hidexdecorations!(axis, label=false, minorgrid=false)
	hideydecorations!(axis, label=false, minorgrid=false)
	xlims!(axis, -sim.params.width/2, +sim.params.width/2)
	ylims!(axis, 0, +sim.params.height)

	positions = Node(sim.positions[:,1])
	scatter!(axis, positions, markersize=2*sim.params.radius, markerspace=AbstractPlotting.SceneSpace, color=:red)

	fps = Int(round(nf / (t2 - t1))) # make duration equal to simulation time in seconds
	record(figure, path, framerate=fps) do io
		for f in 1:nf
			print("\rAnimating frame $f / $nf ...")
			positions[] = sim.positions[:,frames[f]]

			nalive = sum(sim.alive[:,frames[f]])
			t2 = round(sim.times[end], digits=1)
			t = round(sim.times[frames[f]], digits=1)
			R = sim.params.radius
			S = sim.params.spawn_radius
			axis.title = "N = $nalive        t = $t / $t2        R = $R        S = $S"

			recordframe!(io)
		end
	end
	println() # finish progress printing
	return figure
end

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
# TODO: animate underway (i.e. do not store tons of positions)

params = Parameters(
	N = 50,
	T = 2.0,
	width  = 30.0,
	height = 15.0,
	radius = 0.05,
	spawn_radius = 0.10,
	position_spawner = (p, n, t)      -> (isodd(n) ? -p.width/2 : +p.width/2, rand()*p.height/4),
	velocity_spawner = (p, n, t, pos) -> (ang = -pi/6+pi/3*rand()+pi*iseven(n); (4*cos(ang), 4*sin(ang))),
	max_velocity = 4.0,
)
sim = simulate(params, write_trajectories=false)
Profile.clear_malloc_data() # reset profiler stats after one run
sim = simulate(params, write_trajectories=false)
#animate_trajectories(sim; path="anim.mkv", frameskip=10)
#write_trajectories(sim, "trajectories.dat")
