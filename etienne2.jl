using IJulia # avoids gr-related animation errors (https://github.com/JuliaPlots/Plots.jl/issues/3012)
using LinearAlgebra
using Plots
using Printf
gr(dpi=100, fmt=:png);

struct Simulation
	N::Int64

	radius::Float64

	times::AbstractArray
	positions::Array{Tuple{Float64, Float64}, 2}
	velocities::Array{Tuple{Float64, Float64}, 2}

	scatterlines::Array{Float64}
	scatters::Array{Array{Tuple{Int64, Float64}}}
end

STATE_PLOT_SIZE = 400
WIDTH = 15.0
HEIGHT = 15.0
V0 = 5.0

function spawn_velocity(pos)
    #vx = rand(range(5, 6, length=50))
    vx = V0
    vy = rand(range(-0, +0, length=50))
    if pos[1] > 0
        vx *= -1
    end
    return (vx, vy)
end

function position_is_available(pos, positions; sepdist=2*a)
    return minimum(norm(p.-pos) for p in positions) > sepdist
end

function spawn_particles(N; sepdist=0)
    positions = Array{Tuple{Float64, Float64}}(undef, N)
    velocities = Array{Tuple{Float64, Float64}}(undef, N)
    for n in 1:N
        pos = (-100, -100-n) # surely outside, will respawn on death test
        positions[n] = pos
        velocities[n] = spawn_velocity(positions[n])
    end
    return positions, velocities
end

function out_of_bounds(pos)
    x, y = pos[1], pos[2]
    return x < -WIDTH/2 || x > +WIDTH/2 || y > +HEIGHT || y < -50
end

function interact_hard(positions, velocities, dt, acc1s)
    N = length(positions)
    for i in 1:N
        pos1, vel1 = positions[i], velocities[i]
        for j in i+1:N
            pos2, vel2 = positions[j], velocities[j]
            r = norm(pos2 .- pos1)
            if r < 2*a && dot(vel2.-vel1,pos2.-pos1) <= 0
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
        positions[i] = positions[i] .+ velocities[i] .* dt
    end
end

function simulate(N, t, interaction, radius; dt=nothing, nmoving=N)
    mindt = 0.7 * radius / V0 # 0.7 safety factor
    dt = dt == nothing ? mindt : dt
    
    println("V0 = ", V0)
    println("recommended min dt = ", mindt)
    println("dt = ", dt)
    
    times = 0:dt:t
    NT = length(times)
    
    positions, velocities = spawn_particles(N; sepdist=radius)
    
    positions_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)
    velocities_samples = Array{Tuple{Float64, Float64}, 2}(undef, N, NT)

	scatterlines = [3*HEIGHT/3, 2*HEIGHT/3, 1*HEIGHT/3]
	# scatters = []
	scatters = [[] for i in 1:length(scatterlines)]
    
    function sample(iter, positions, velocities)
        positions_samples[:, iter] = positions
        velocities_samples[:, iter] = velocities
    end
    
    sample(1, positions, velocities)
    
    acc1s = Array{Tuple{Float64, Float64}}(undef, N)
    
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
            if out_of_bounds(positions[n])
                # respawn
                if mod(n, 2) == 0
                    x = -WIDTH/2
                    y = (HEIGHT/5 - 0) * n/N
                else
                    x = +WIDTH/2
                    y = (HEIGHT/5 - 0) * (n+1)/N
                end
                pos = (x, y)
                
                if position_is_available(pos, positions, sepdist=2.5*radius)
                    x, y = positions[n][1], positions[n][2]
                    
                    positions[n] = pos
                    velocities[n] = spawn_velocity(pos)
                    side = mod1(side + 1, 2) # spawn on other side next time
                else
                    # then try respawning again at next time step (don't force it!)
                end
            end
        end
        sample(iter, positions, velocities)
    end
    println()
    
    return Simulation(N, radius, times, positions_samples, velocities_samples, scatterlines, scatters)
end

function plot_geometry(p, scatterlines=nothing)
	if scatterlines != nothing
		hline!(p, scatterlines, color=:black, lw=2, alpha=0.25)
	end
end

function plot_trajectories(times, trajectories)
    p = plot_state(times[1], trajectories[:, 1])
    p = plot(p, title="Trajectories")

    N, T = size(trajectories)
    for n in 1:N
        trajectory = trajectories[n, :]
        plot!(p, trajectory, arrow=:arrow, color=n)
    end
    return p
end

# function plot_state(time, positions; velocities=nothing, scatterlines=nothing)
function plot_state(sim::Simulation, i; velocities=nothing, scatterlines=nothing)
	time, positions = sim.times[i], sim.positions[:,i]

    N = size(positions)[1]
    title = @sprintf("State for N = %d at t = %.3f", N, time)
    p = plot(title=title, xlim=(-WIDTH/2, +WIDTH/2), ylim=(0, HEIGHT), size=(STATE_PLOT_SIZE, STATE_PLOT_SIZE), legend=nothing, xlabel="x", ylabel="y")
    plot_geometry(p, scatterlines)
    scatter!(p, positions, color=1:N, markersize=STATE_PLOT_SIZE * sim.radius/(WIDTH))
    if velocities != nothing
        for n in 1:N
            pos, vel = positions[n], velocities[n]
            plot!(p, [pos, pos .+ vel], arrow=:arrow, color=n)
        end
    end
    return p
end

function plot_scatters(scatters_so_far)
	N = length(scatters_so_far)
	p = plot(layout=(N, 1))
	for i in 1:N
		if length(scatters_so_far[i]) > 0
			p = histogram!(p, scatters_so_far[i], xlim=(-WIDTH/2, +WIDTH/2), bins=range(-WIDTH/2, stop=+WIDTH/2, length=15), normalized=true, title="Scattering distribution", legend=nothing, subplot=i)
		else
			p = plot!(p)
		end
	end
	return p
end

# function animate_trajectories(times, trajectories, scatterlines, scatters, dt; t=nothing, fps=30)
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

        # p2 = plot_state(sim.times[t], trajectories[:, t], scatterlines=sim.scatterlines)
        p2 = plot_state(sim, t, scatterlines=sim.scatterlines)

		for j in 1:length(sim.scatters)
			while scatteri[j] < length(sim.scatters[j]) && sim.scatters[j][scatteri[j]][1] <= t
				push!(scatters_so_far[j], sim.scatters[j][scatteri[j]][2])
				scatteri[j] += 1
			end
		end

		p1 = plot_scatters(scatters_so_far)
        plot(p1, p2, layout=(2, 1), size=(STATE_PLOT_SIZE, STATE_PLOT_SIZE*2))
    end
    println()
    return mp4(anim, "anim.mp4", fps=fps)
end

function total_energy(positions, velocities)
    N, T = size(positions)
    energies = Array{Float64}(undef, T)
    for t in 1:T
        energies[t] = 0
        for n in 1:N
            v = norm(velocities[n, t])
            kin = 1/2*m*v^2
            pos = positions[n, t]

            # Lennard-Jones
            pot = potential_wall(pos) + 1/2 * potential_parts(n, positions[:, t])

            energies[t] += kin + pot
        end
    end
    return energies
end

function plot_energy(times, positions, velocities)
    energies = total_energy(positions, velocities)
    p = plot(title="Energy", times, energies, ylim=(0, 2*maximum(energies)), label="", xlabel="t", ylabel="E")
    return p
end

function plot_triple(times, positions, velocities)
    p1 = plot_state(times[1], positions[:, 1], velocities=velocities[:, 1])
    p2 = plot_trajectories(times, positions)
    p3 = plot_energy(times, positions, velocities)
    p = plot(p1, p2, p3, size=(1100, 350), layout=(1, 3), bottom_margin=8*Plots.mm)
    return p
end

# times, positions, velocities, scatterlines, scatters = simulate(250, 30, interact_hard, dt=0.01)#; dt=0.005)
# plot_triple(times, positions, velocities)
# animate_trajectories(times, positions, scatterlines, scatters, 0.1, fps=20)

sim = simulate(50, 10, interact_hard, 0.2, dt=0.01)
animate_trajectories(sim, dt=0.1, fps=20)
