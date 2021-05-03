using LinearAlgebra
using Printf
using Javis
using ProgressMeter
using GR

@Base.kwdef struct Parameters # allow construction with Parameters(N=...)
	N::Int
	T::Float64

	width::Float64
	height::Float64

	radius::Float64
	spawn_separation::Float64

	spawn_ymin::Float64
	spawn_ymax::Float64
	spawn_vmin::Float64
	spawn_vmax::Float64
	spawn_angmin::Float64
	spawn_angmax::Float64
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

struct Spawner
	# position
	x::Float64
	ymin::Float64
	ymax::Float64

	# velocity
	vmin::Float64
	vmax::Float64
	angmin::Float64
	angmax::Float64
end

function spawn(spawner::Spawner)
	x = spawner.x
	y = spawner.ymin + rand() * (spawner.ymax - spawner.ymin)
	pos = (x, y)

	v = spawner.vmin + rand() * (spawner.vmax - spawner.vmin)
	ang = spawner.angmin + rand() * (spawner.angmax - spawner.angmin)
	vel = (v * cos(ang), v * sin(ang))

	return pos, vel
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

function simulate(params)
    dt = 0.1 * params.radius / params.spawn_vmax # 0.7 safety factor # 1.0 would mean particle centers could overlap in one step
    times = 0:dt:params.T
    NT = length(times)
    
    positions = Array{Tuple{Float64, Float64}}(undef, params.N)
    velocities = Array{Tuple{Float64, Float64}}(undef, params.N)
	alive = fill(false, params.N)
    
    positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, params.N, NT)
    velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, params.N, NT)
    alive_samples = Array{Bool, 2}(undef, params.N, NT)

	# aim for 1 particle in each cell, so one particle can be in at most 4 cells
	ncellsx = Int(floor(params.width / (2*params.radius))-1) # floor & reduce, so at most 4 particles in each cell
	ncellsy = Int(floor(params.height / (2*params.radius))-1)
	cellwidth = params.width / ncellsx
	cellheight = params.height / ncellsy

	max_parts_per_cell = Int(round(9*cellwidth*cellheight / (pi*params.radius^2))) # assume complete filling for simple upper bound

	# TODO: use some form of map type?
	# maps from cell -> particle and particle -> cell
	celllen = zeros(Int, (ncellsx, ncellsy))
	cell2part = Array{Tuple{Int, Int}, 3}(undef, ncellsx, ncellsy, max_parts_per_cell) # (cx, cy, ci) -> (particle id, cell #1-4)
	part2cell = Array{Tuple{Int, Int, Int}, 2}(undef, params.N, 4) # (particle id, cell #1-4) -> (cx, cy, ci)

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
		log_trajectory(n, iter)
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

	spawnerl = Spawner(-params.width/2, params.spawn_ymin, params.spawn_ymax, params.spawn_vmin, params.spawn_vmax, 0  + params.spawn_angmin, 0  + params.spawn_angmax)
	spawnerr = Spawner(+params.width/2, params.spawn_ymin, params.spawn_ymax, params.spawn_vmin, params.spawn_vmax, pi + params.spawn_angmin, pi + params.spawn_angmax)

	function spawn_particle(n, iter, pos, vel)
		alive[n] = true
		positions[n] = pos
		velocities[n] = vel
		addpart(n) # add to cells
		start_trajectory(n)
		log_trajectory(n, iter)
	end

	function position_is_available(pos1)
		cx1, cy1, cx2, cy2 = pos2cells(pos1, params.width, params.height, ncellsx, ncellsy)
		for cx in cx1:cx2 # TODO: create some form of cleaner iteration
			for cy in cy1:cy2
				for i in 1:celllen[cx,cy]
					n2 = cell2part[cx,cy,i][1]
					pos2 = positions[n2]
					if norm(pos2 .- pos1) < params.spawn_separation
						return false
					end
				end
			end
		end
		return true
	end
    
    @showprogress 1 "Simulating..." for iter in 1:NT
		# kill dead particles and (try to) respawn them
		for n1 in 1:params.N
            if alive[n1] && out_of_bounds(params.width, params.height, positions[n1])
				kill_particle(n1, iter)
			end
			if !alive[n1]
				pos, vel = spawn(n1 % 2 == 0 ? spawnerl : spawnerr)
                if position_is_available(pos)
					spawn_particle(n1, iter, pos, vel)
                end
            end

			# sample current positions and velocities
			positions_samples[n1,iter] = positions[n1]
			velocities_samples[n1,iter] = velocities[n1]
			alive_samples[n1,iter] = alive[n1]

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

				if has_scattered
					log_trajectory(n1, iter) # log trajectory only when scattering
				end
			end
		end
    end
    
    return Simulation(params, times, positions_samples, velocities_samples, alive_samples, trajectories, ncellsx, ncellsy)
end

function animate_trajectories_javis(sim::Simulation; fps=30, path="anim.mp4", frameskip=1, interactive=false)
	frames = 1:frameskip:length(sim.times)
	nf = length(frames)

	function ground(args...)
		background("white")
		sethue("black")
	end

	WIDTH = 1000
	HEIGHT = sim.params.height / sim.params.width * WIDTH

	world2canvasr(r) = r / sim.params.width * WIDTH
	world2canvasx(x) = -WIDTH/2 + WIDTH * (x - -sim.params.width/2) / sim.params.width
	world2canvasy(y) = -HEIGHT/2 + HEIGHT * (sim.params.height - y) / sim.params.height
	world2canvasxy(xy) = (world2canvasx(xy[1]), world2canvasy(xy[2]))

	function object(pt, color)
		circle(pt, world2canvasr(sim.params.radius), :fill)
		return pt
	end

	function textlabel(str)
		fontsize(20)
		text(str, Point(-0.98*WIDTH/2, -0.98*HEIGHT/2); valign=:top)
		return str
	end

	function draw(video, action, frame)
		nalive = sum(sim.alive[:,frames[frame]])
		alive_particles = Iterators.filter(n -> sim.alive[n,frames[frame]], 1:sim.params.N)
		return vcat(
			[object(Point(world2canvasxy(sim.positions[n,frames[frame]])), "black") for n in alive_particles],
			[textlabel("N = $nalive")],
		)
	end

	video = Video(WIDTH, HEIGHT)
	javis(video, [
		BackgroundAction(1:nf, ground), 
		Action(1:nf, :red_ball, draw),
	], pathname=path, liveview=interactive)
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

# TODO: is it randomized spawning position or velocity that causes symmetry breaking?
# TODO: animate underway (i.e. do not store tons of positions)
# TODO: output trajectories
# TODO: let user write own spawning functions?

params = Parameters(
	N = 200,
	T = 20.0,
	width  = 30.0,
	height = 15.0,
	radius = 0.1,
	spawn_separation = 0.2,
	spawn_ymin = 0.0,
	spawn_ymax = 2.0,
	spawn_vmin = 2.0, 
	spawn_vmax = 3.0,
	spawn_angmin = -pi/6, 
	spawn_angmax = +pi/6,
)
sim = simulate(params)
# animate_trajectories_javis(sim; fps=30, path="anim.mp4", frameskip=5, interactive=true)
