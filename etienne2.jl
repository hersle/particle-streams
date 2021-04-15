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

	scatterlines::Array{Float64}
	scatters::Array{Array{Tuple{Int64, Float64}}}
end

function spawn_position(width, height, n, N)
	# respawn
	if mod(n, 2) == 0
		return (-width/2, (height/5 - 0) * (n+0)/N)
	else
		return (+width/2, (height/5 - 0) * (n+1)/N)
	end
	return (x, y)
end

function spawn_velocity(pos, v0)
	if pos[1] < 0
		return (+v0, 0) # spawn on left, move to right
	else
		return (-v0, 0) # spawn on right, move to left
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

function simulate(N, t, radius, width, height, v0)
    dt = 0.1 * radius / v0 # 0.7 safety factor # 1.0 would mean particle centers could overlap in one step
    
    println("v0 = ", v0)
    println("dt = ", dt)
    
    times = 0:dt:t
    NT = length(times)
    
    positions, velocities = spawn_particles(N, width, height, radius)
    
    positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)
    velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)

	scatterlines = [3*height/3, 2*height/3, 1*height/3]
	scatters = [[] for i in 1:length(scatterlines)]
    
    function sample(iter, positions, velocities)
        positions_samples[:, iter] = positions
        velocities_samples[:, iter] = velocities
    end
    
    sample(1, positions, velocities)
    
    side = 1
    
    for iter in 2:NT # remaining NT - 1 iterations
        if iter % Int(round(NT / 40, digits=0)) == 0 || iter == NT
            print("\rSimulating $N particle(s) in $NT time steps: $(Int(round(iter/NT*100, digits=0))) %")
        end
                
		for i in 1:N
			pos1, vel1 = positions[i], velocities[i]
			for j in i+1:N
				pos2, vel2 = positions[j], velocities[j]
				r = norm(pos2 .- pos1)
				if r < 2*radius && dot(vel2.-vel1,pos2.-pos1) <= 0
					dvel1 = dot(vel1.-vel2,pos1.-pos2) / (r*r) .* (pos1 .- pos2)
					vel1, vel2 = vel1 .- dvel1, vel2 .+ dvel1
					velocities[i] = vel1
					velocities[j] = vel2
				end
			end
		end
		for i in 1:N
			if positions[i][2] < 0 && velocities[i][2] < 0
				velocities[i] = velocities[i] .* (+1, -1)
			end
		end
		for i in 1:N
			newpos = positions[i] .+ velocities[i] .* dt

			# register scattering
			for j in 1:length(scatterlines)
				scatterliney = scatterlines[j]
				if positions[i][2] < scatterliney && newpos[2] >= scatterliney
					push!(scatters[j], (iter, newpos[1]))
				end
			end

			positions[i] = newpos
		end

        for n in 1:N
            if out_of_bounds(width, height, positions[n])
				pos = spawn_position(width, height, n, N)
                if position_is_available(pos, positions, sepdist)
                    positions[n] = pos
                    velocities[n] = spawn_velocity(pos, v0)
                    side = mod1(side + 1, 2) # spawn on other side next time
                else
                    # then try respawning again at next time step (don't force it!)
                end
            end
        end
        sample(iter, positions, velocities)
    end
    println() # end progress writer
    
    return Simulation(N, width, height, radius, times, positions_samples, velocities_samples, scatterlines, scatters)
end

function plot_state(sim::Simulation, i; velocity_scale=0.0, scatterlines=nothing)
	time, positions, velocities = sim.times[i], sim.positions[:,i], sim.velocities[:,i]

    N = size(positions)[1]
    title = @sprintf("State for N = %d at t = %.3f", N, time)
    p = plot(title=title, xlim=(-sim.width/2, +sim.width/2), ylim=(0, sim.height), size=(STATE_PLOT_SIZE, STATE_PLOT_SIZE), legend=nothing, xlabel="x", ylabel="y")
	if scatterlines != nothing
		hline!(p, scatterlines, color=:black, lw=2, alpha=0.25)
	end
    scatter!(p, positions, color=1:N, markersize=STATE_PLOT_SIZE * sim.radius/(sim.width))
    if velocity_scale != 0.0
        for n in 1:N
            pos, vel = positions[n], velocities[n]
            plot!(p, [pos, pos .+ vel .* velocity_scale], arrow=:arrow, color=n)
        end
    end
    return p
end

function plot_scatters(sim::Simulation, scatters_so_far)
	N = length(scatters_so_far)
	p = plot(layout=(N, 1))
	for i in 1:N
		if length(scatters_so_far[i]) > 0
			p = histogram!(p, scatters_so_far[i], xlim=(-sim.width/2, +sim.width/2), bins=range(-sim.width/2, stop=+sim.width/2, length=15), normalized=true, title="Scattering distribution", legend=nothing, subplot=i)
		else
			p = plot!(p)
		end
	end
	return p
end

function animate_trajectories(sim::Simulation; dt=nothing, t=nothing, fps=30)
	dt = dt == nothing ? sim.times[2]-sim.times[1] : dt
    t = t == nothing ? sim.times[end] : t
    f = Int(round(t / (sim.times[2] - sim.times[1]), digits=0))
    skip = Int(round(dt / (sim.times[2] - sim.times[1]), digits=0))
    println("skip:", skip)

    trajectories = sim.positions[:,1:f] # respect upper time
    #times = times[:Int(round(t/dt, digits=0))]

    N, T = size(trajectories)
    scatters_so_far = [[] for i in 1:length(sim.scatters)]
	scatteri = [1 for i in 1:length(sim.scatters)]
    anim = @animate for t in 1:skip:T
        if true
            print("\rAnimating $N particle(s) in $(length(1:skip:T)) time steps: $(Int(round(t/T*100, digits=0))) %")
        end

        p2 = plot_state(sim, t, velocity_scale=0.25, scatterlines=sim.scatterlines)

		for j in 1:length(sim.scatters)
			while scatteri[j] < length(sim.scatters[j]) && sim.scatters[j][scatteri[j]][1] <= t
				push!(scatters_so_far[j], sim.scatters[j][scatteri[j]][2])
				scatteri[j] += 1
			end
		end

		p1 = plot_scatters(sim, scatters_so_far)
        plot(p1, p2, layout=(2, 1), size=(STATE_PLOT_SIZE, STATE_PLOT_SIZE*2))
    end
    println()
    return mp4(anim, "anim.mp4", fps=fps)
end

sim = simulate(50, 10, 0.2, 15.0, 15.0, 5.0)
animate_trajectories(sim, dt=0.1, fps=20)
