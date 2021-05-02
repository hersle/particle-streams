using LinearAlgebra
using Printf
using Javis

struct Simulation
	N::Int64

	width::Float64
	height::Float64
	radius::Float64

	times::AbstractArray
	positions::Array{Tuple{Float64, Float64}, 2}
	velocities::Array{Tuple{Float64, Float64}, 2}
	nalive::Array{Int}

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

function spawn_particles(N, width, height, sepdist)
    positions = Array{Tuple{Float64, Float64}}(undef, N)
    velocities = Array{Tuple{Float64, Float64}}(undef, N)
    for n in 1:N
        positions[n] = (-100*width, -(100+4*n*sepdist)) # surely outside, will respawn on death test
        velocities[n] = (0, 0) # irrelevant, will respawn immediately anyway
    end
    return positions, velocities
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
	end
	return vel1, vel2
end

function scatter_wall(pos, vel, radius)
	if pos[2] < 0 && vel[2] < 0
		return (vel[1], -vel[2]) # reflect y-velocity
	end
	return vel
end

# TODO: is it randomized spawning position or velocity that causes symmetry breaking?
function simulate(N, t, radius, width, height, v0, sepdistmult, spawnymax, spawnvelang)
    dt = 0.1 * radius / v0 # 0.7 safety factor # 1.0 would mean particle centers could overlap in one step
    
    println("v0 = ", v0)
    println("dt = ", dt)
    
    times = 0:dt:t
    NT = length(times)

	sepdist = sepdistmult * radius
    
    positions, velocities = spawn_particles(N, width, height, radius)
    
    positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)
	println(size(positions_samples))
    velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)
	nalive_samples = Array{Int}(undef, NT)

	nalive = 0
	alive = fill(false, N)
    
    function sample(iter)
        positions_samples[:, iter] = positions
        velocities_samples[:, iter] = velocities
		nalive_samples[iter] = nalive
    end
    
    sample(1)

	ncellsx = Int(floor(width / radius)-1) # floor & reduce, so at most 4 particles in each cell
	ncellsy = Int(floor(height / radius)-1)
	cellwidth = width / ncellsx
	cellheight = height / ncellsy

	println("ncells: ($ncellsx, $ncellsy)")
	println("radius: $radius")
	println("cell size: ($cellwidth, $cellheight)")

	# TODO: fix this
	max_parts_per_cell = Int(round(2 * 9 * cellwidth * cellheight / (pi*radius^2))) # assume complete filling

	# maps from cell -> particle and particle -> cell
	celllen = Array{Int, 2}(undef, ncellsx, ncellsy)
	cell2part = Array{Tuple{Int, Int}, 3}(undef, ncellsx, ncellsy, max_parts_per_cell) # (cx, cy, ci) -> (particle id, cell #1-4)
	part2cell = Array{Tuple{Int, Int, Int}, 2}(undef, N, 4) # (particle id, cell #1-4) -> (cx, cy, ci)
	for cx in 1:ncellsx
		for cy in 1:ncellsy
			celllen[cx,cy] = 0
		end
	end

	function addpart(n)
		pos = positions[n]
		cx1, cy1, cx2, cy2 = pos2cells(pos, width, height, ncellsx, ncellsy)
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

	function kill_particle(n)
		rempart(n)
		alive[n] = false
		nalive -= 1
	end

	spawnerl = Spawner(-width/2, 0*height, 0.1*height, v0, 3*v0, -spawnvelang/2, +spawnvelang/2)
	spawnerr = Spawner(+width/2, 0*height, 0.1*height, v0, v0, -spawnvelang/2+pi, +spawnvelang/2+pi)

	function spawn_particle(n, pos, vel)
		alive[n] = true
		nalive += 1
		positions[n] = pos
		velocities[n] = vel
		addpart(n) # add to cells
	end

	function position_is_available(pos1)
		cx1, cy1, cx2, cy2 = pos2cells(pos1, width, height, ncellsx, ncellsy)
		for cx in cx1:cx2 # TODO: create some form of cleaner iteration
			for cy in cy1:cy2
				for i in 1:celllen[cx,cy]
					n2 = cell2part[cx,cy,i][1]
					pos2 = positions[n2]
					if norm(pos2 .- pos1) < sepdist
						return false
					end
				end
			end
		end
		return true
	end
    
    for iter in 2:NT # remaining NT - 1 iterations
		progress = Int(round(iter/NT * 100))
		print("\rSimulating timestep $iter/$NT ($progress %) ...")

		for n in 1:N
			if alive[n]
				rempart(n) # remove from current cell (based on prev pos)
				addpart(n) # add to next cell (based on current pos)
			end
		end

		for n1 in 1:N
			if alive[n1]
				pos1, vel1 = positions[n1], velocities[n1]
				cx1, cy1, cx2, cy2 = pos2cells(pos1, width, height, ncellsx, ncellsy)
				for cx in cx1:cx2 # TODO: create some form of cleaner iteration
					for cy in cy1:cy2
						for i in 1:celllen[cx,cy]
							n2 = cell2part[cx,cy,i][1]
							pos2, vel2 = positions[n2], velocities[n2]
							if alive[n2] && n2 > n1
								vel1, vel2 = scatter_particles(pos1, vel1, pos2, vel2, radius) # velocities[n1], velocities[n2] = scatter_particles() causes errors!
								velocities[n1], velocities[n2] = vel1, vel2
							end
						end
					end
				end

				vel1 = scatter_wall(pos1, vel1, radius)
				velocities[n1] = vel1
			end
		end
                
		for i in 1:N
			newpos = positions[i] .+ velocities[i] .* dt
			positions[i] = newpos
		end

        for n in 1:N
            if alive[n] && out_of_bounds(width, height, positions[n])
				kill_particle(n)
			end
			if !alive[n]
				if n % 2 == 0
					pos, vel = spawn(spawnerl)
				else
					pos, vel = spawn(spawnerr)
				end
                if position_is_available(pos)
					spawn_particle(n, pos, vel)
                end
            end
        end
        sample(iter)
    end
    println() # end progress writer
    
    return Simulation(N, width, height, radius, times, positions_samples, velocities_samples, nalive_samples, ncellsx, ncellsy)
end

function animate_trajectories_javis(sim::Simulation; fps=30, path="anim.mp4", frameskip=1)
	frames = 1:frameskip:length(sim.times)
	nf = length(frames)

	function ground(args...)
		background("white")
		sethue("black")
	end

	WIDTH = 1000
	HEIGHT = sim.height / sim.width * WIDTH

	world2canvasr(r) = r / sim.width * WIDTH
	world2canvasx(x) = -WIDTH/2 + WIDTH * (x - -sim.width/2) / sim.width
	world2canvasy(y) = -HEIGHT/2 + HEIGHT * (sim.height - y) / sim.height
	world2canvasxy(xy) = (world2canvasx(xy[1]), world2canvasy(xy[2]))

	function object(pt, color)
		circle(pt, world2canvasr(sim.radius), :fill)
		return pt
	end

	function textlabel(str)
		fontsize(20)
		text(str, Point(-0.98*WIDTH/2, -0.98*HEIGHT/2); valign=:top)
		return str
	end

	function draw(video, action, frame)
		return vcat(
			[object(Point(world2canvasxy(sim.positions[n,frames[frame]])), "black") for n in 1:sim.N],
			[textlabel("N = $(sim.nalive[frames[frame]])")],
		)
	end

	video = Video(WIDTH, HEIGHT)
	javis(video, [
		BackgroundAction(1:nf, ground), 
		Action(1:nf, :red_ball, draw),
	], pathname=path)
end

sim = simulate(50, 5, 0.1, 30.0, 15.0, 5.0, 2.1, 5, pi/6)
animate_trajectories_javis(sim; fps=30, path="anim.mp4", frameskip=10)
