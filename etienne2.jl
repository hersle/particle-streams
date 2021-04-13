using IJulia # avoids gr-related animation errors (https://github.com/JuliaPlots/Plots.jl/issues/3012)
using LinearAlgebra
using Plots
using Printf
gr(dpi=100, fmt=:png);

@enum Interaction HARD SOFT
INTERACTION = HARD

a = 0.2
m = 1.0
#R = 5.0
K = 25.0
ϵ = 10.0

STATE_PLOT_SIZE = 400
WIDTH = 15.0
HEIGHT = 15.0
V0 = 5.0

function force_wall(pos)
    y = pos[2]
    if pos[2] < 0
        return K * -1 .* (0, y)
    else
        return (0, 0)
    end
end

function potential_wall(pos)
    y = pos[2]
    if y > 0
        return 0
    else
        return K/2 * y^2
    end
end

function force_part(i, j, positions)
    rij = positions[j] .- positions[i]
    r = norm(rij)
    if i == j || r > a
        return (0, 0)
    else
        return (-12 * ϵ * ((a/r)^12 - (a/r)^6) / r^2) .* rij
    end
end

function force_parts(i, positions)
    N = length(positions)
    force = (0, 0)
    for j in 1:N
        force = force .+ force_part(i, j, positions)
    end
    return force
end

function potential_part(i, j, positions)
    rij = positions[j] .- positions[i]
    r = norm(rij)
    if i == j || r > a
        return 0
    else
        return ϵ * ((a/r)^12 - 2*(a/r)^6 + 1)
    end
end

function potential_parts(i, positions)
    N = length(positions)
    potential = sum(potential_part(i, j, positions) for j in 1:N)
    return potential
end


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

function interact_hard_symmetric(positions, velocities, dt, acc1s)
    # TODO: forget this, the above one works?
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

function interact_soft(positions, velocities, dt, acc1s)
    N = length(positions)
    for n in 1:N
        pos, vel = positions[n], velocities[n]
        force1 = force_wall(pos) .+ force_parts(n, positions)
        acc1 = force1 ./ m
        acc1s[n] = acc1
        pos = pos .+ vel .* dt .+ 1/2 .* acc1 .* dt^2
        positions[n] = pos
    end
    for n in 1:N
        pos, vel = positions[n], velocities[n]
        force2 = force_wall(pos) .+ force_parts(n, positions)
        acc2 = force2 ./ m
        acc1 = acc1s[n]
        acc = (acc1 .+ acc2) ./ 2
        vel = vel .+ acc .* dt
        velocities[n] = vel
    end
end

function simulate(N, t, interaction; dt=nothing, nmoving=N)
    mindt = 0.7 * a / V0 # 0.7 safety factor
    dt = dt == nothing ? mindt : dt
    
    println("V0 = ", V0)
    println("recommended min dt = ", mindt)
    println("dt = ", dt)
    
    times = 0:dt:t
    NT = length(times)
    
    positions, velocities = spawn_particles(N; sepdist=a)
    
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
                # pos = spawn_position(false)
                if mod(n, 2) == 0
                    x = -WIDTH/2
                    y = (HEIGHT/5 - 0) * n/N
                else
                    x = +WIDTH/2
                    y = (HEIGHT/5 - 0) * (n+1)/N
                end
                pos = (x, y)
                
                if position_is_available(pos, positions, sepdist=2.5*a)
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
    
    return times, positions_samples, velocities_samples, scatterlines, scatters
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

function plot_state(time, positions; velocities=nothing, scatterlines=nothing)
    N = size(positions)[1]
    title = @sprintf("State for N = %d at t = %.3f", N, time)
    p = plot(title=title, xlim=(-WIDTH/2, +WIDTH/2), ylim=(0, HEIGHT), size=(STATE_PLOT_SIZE, STATE_PLOT_SIZE), legend=nothing, xlabel="x", ylabel="y")
    plot_geometry(p, scatterlines)
    scatter!(p, positions, color=1:N, markersize=STATE_PLOT_SIZE * a/(WIDTH))
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

function animate_trajectories(times, trajectories, scatterlines, scatters, dt; t=nothing, fps=30)
    t = t == nothing ? times[end] : t
    f = Int(round(t / (times[2] - times[1]), digits=0))
    skip = Int(round(dt / (times[2] - times[1]), digits=0))
    println("skip:", skip)

    trajectories = trajectories[:,1:f] # respect upper time
    #times = times[:Int(round(t/dt, digits=0))]

    N, T = size(trajectories)
    scatters_so_far = [[] for i in 1:length(scatters)]
	scatteri = [1 for i in 1:length(scatters)]
    anim = @animate for t in 1:skip:T
        if true
            print("\rAnimating $N particle(s) in $(length(1:skip:T)) time steps: $(Int(round(t/T*100, digits=0))) %")
        end

        p2 = plot_state(times[t], trajectories[:, t], scatterlines=scatterlines)

		for j in 1:length(scatters)
			while scatteri[j] < length(scatters[j]) && scatters[j][scatteri[j]][1] <= t
				push!(scatters_so_far[j], scatters[j][scatteri[j]][2])
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

times, positions, velocities, scatterlines, scatters = simulate(250, 30, interact_hard, dt=0.01)#; dt=0.005)
plot_triple(times, positions, velocities)
animate_trajectories(times, positions, scatterlines, scatters, 0.1, fps=20)
