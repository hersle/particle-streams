using IJulia # avoids gr-related animation errors (https://github.com/JuliaPlots/Plots.jl/issues/3012)
using LinearAlgebra
using Plots
using Printf
gr(dpi=100, fmt=:png);

STATE_PLOT_SIZE = 400

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

function spawn_position(width, height, n, N, spawnymax)
	# respawn
	if mod(n, 2) == 0
		x = -width/2
	else
		x = +width/2
	end
	y = rand() * spawnymax
	return (x,y)
end

function spawn_velocity(pos, v0, spawnvelang)
	ang = -spawnvelang/2 + spawnvelang * rand()
	if pos[1] < 0
		return (+v0*cos(ang), v0*sin(ang)) # spawn on left, move to right
	else
		return (-v0*cos(ang), v0*sin(ang)) # spawn on right, move to left
	end
end

function position_is_available(pos, positions, sepdist)
    return minimum(norm(p.-pos) for p in positions) > sepdist
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
    velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)
	nalive_samples = Array{Int}(undef, NT)

    side = 1 # which side to spawn on
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

	println("ncells: ($ncellsx, $ncellsy)")
	println("radius: $radius")
	println("cell size: ($(width/ncellsx), $(height/ncellsy))")

	# maps from cell -> particle and particle -> cell
	celllen = Array{Int, 2}(undef, ncellsx, ncellsy)
	cell2part = Array{Tuple{Int, Int}, 3}(undef, ncellsx, ncellsy, N) # (cx, cy, ci) -> (particle id, cell #1-4)
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
		alive[n] = false
		nalive -= 1
	end

	function spawn_particle(n, pos, vel)
		alive[n] = true
		nalive += 1
		positions[n] = pos
		velocities[n] = vel
		side = mod1(side + 1, 2) # spawn on other side next time

		addpart(n) # add to cells
	end
    
    for iter in 2:NT # remaining NT - 1 iterations
        if iter % Int(round(NT / 40, digits=0)) == 0 || iter == NT
            print("\rSimulating $N particle(s) in $NT time steps: $(Int(round(iter/NT*100, digits=0))) %")
        end

		for n in 1:N
			if alive[n]
				rempart(n)
				addpart(n)
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
				pos = spawn_position(width, height, n, N, spawnymax)
				vel = spawn_velocity(pos, v0, spawnvelang)
                if position_is_available(pos, positions, sepdist)
					spawn_particle(n, pos, vel)
                end
            end
        end
        sample(iter)
    end
    println() # end progress writer
    
    return Simulation(N, width, height, radius, times, positions_samples, velocities_samples, nalive_samples, ncellsx, ncellsy)
end

function plot_state(sim::Simulation, i; velocity_scale=0.0, grid=false)
	time, positions, velocities = sim.times[i], sim.positions[:,i], sim.velocities[:,i]

    N = size(positions)[1]
    title = @sprintf("State for N = %d at t = %.3f", sim.nalive[i], time)
    p = plot(title=title, xlim=(-sim.width/2, +sim.width/2), ylim=(0, sim.height), size=(STATE_PLOT_SIZE, sim.height/sim.width*STATE_PLOT_SIZE), legend=nothing, xlabel="x", ylabel="y")
    scatter!(p, positions, color=:black, markersize=STATE_PLOT_SIZE * sim.radius/(sim.width))
    if velocity_scale != 0.0
        for n in 1:N
            pos, vel = positions[n], velocities[n]
            plot!(p, [pos, pos .+ vel .* velocity_scale], arrow=:arrow, color=n)
        end
    end
	if grid
		p = plot!(p, xticks=range(-sim.width/2,+sim.width/2,length=sim.ncellsx+1), yticks=range(0,sim.height,length=sim.ncellsy+1), grid=true)
	end
    return p
end

# TODO: use plotting library that shows correct radius
function animate_trajectories(sim::Simulation; velocity_scale=0.0, dt=nothing, t=nothing, fps=30, path="anim.mp4")
	dt = dt == nothing ? sim.times[2]-sim.times[1] : dt
    t = t == nothing ? sim.times[end] : t
    f = Int(round(t / (sim.times[2] - sim.times[1]), digits=0))
    skip = Int(round(dt / (sim.times[2] - sim.times[1]), digits=0))
    println("skip:", skip)

    trajectories = sim.positions[:,1:f] # respect upper time
    #times = times[:Int(round(t/dt, digits=0))]

    N, T = size(trajectories)
    anim = @animate for t in 1:skip:T
        if true
            print("\rAnimating $N particle(s) in $(length(1:skip:T)) time steps: $(Int(round(t/T*100, digits=0))) %")
        end

        p2 = plot_state(sim, t; velocity_scale=velocity_scale, grid=false)
    end
    println()
    anim = mp4(anim, "anim.mp4", fps=fps)
	run(`ffmpeg -y -i anim.mp4 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p anim_fixed.mp4`) # convert
	run(`mv anim_fixed.mp4 $path`)
	return anim
end

sim = simulate(40, 10, 0.5, 30.0, 15.0, 5.0, 2.1, 5, pi/6)
animate_trajectories(sim, dt=0.01, fps=20, velocity_scale=0.00, path="anim.mp4")
