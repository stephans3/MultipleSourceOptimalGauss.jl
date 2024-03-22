
using Hestia 
θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 400.0   # Specific heat capacity: constant

L = 0.3     # Length
W = 0.02     # Width

Nx = 30     # Number of elements: x direction
Ny = 5     # Number of elements: y direction
Ntotal = Nx*Ny

property = StaticIsotropic(λ, ρ, c)
plate    = HeatPlate(L, W, Nx, Ny)

### Boundaries ###
θamb = 300.0;
emission = Emission(10, θamb, 0.1)   # Convection and Radiation

boundary  = Boundary(plate)
setEmission!(boundary, emission, :west)
setEmission!(boundary, emission, :east)
setEmission!(boundary, emission, :north)

### Actuation ###
# Create actuator characterization
scale     = 1.0;
power     = 3;
curvature = 20.0;
num_actuators = 3;
config_a  = RadialCharacteristics(scale, power, curvature) 
actuation = IOSetup(plate)
setIOSetup!(actuation, plate, 3, config_a,  :south)

### Sensor ###
# Create sensor characterization
scale     = 1.0;
power     = 2;
curvature = 20.0;
num_sensor = 3;
config_s  = RadialCharacteristics(scale, power, curvature)
sensing   = IOSetup(plate)
setIOSetup!(sensing, plate, num_sensor, config_s,  :north)


# Parametrized input function
function input_signal_oc(time,p₁,p₂,p₃)
    return  exp(p₁- p₃^2 * (time - 1/p₂)^2)
end

# Model Predictive Control
# Quasi-continuous steps
function step_gauss(t, T, n; init_value = 1e-6, power=10)
    m = -log(init_value)
    f(t) = -m*(2*(t/T + 0.5 - n ))^(2*power)
    return exp(f(t))
end

function mpc_signal(t, T, vec,iter)
    s = 0.0;
    for (id, v) in enumerate(vec) 
        s += v * step_gauss(t,T, id+iter)
    end
    return s
end

# Heat Equation
function heat_conduction!(dθ, θ, p_mpc, t, iter; p_oc = zeros(9))
    time = t/Toc 
    u1_oc = input_signal_oc(time,p_oc[1:3]...)
    u2_oc = input_signal_oc(time,p_oc[4:6]...)
    u3_oc = input_signal_oc(time,p_oc[7:9]...)
    
    t_mpc = t-T_offset;
    u1_mpc = mpc_signal(t_mpc, dt_mpc, p_mpc[1,:],iter)
    u2_mpc = mpc_signal(t_mpc, dt_mpc, p_mpc[2,:],iter)
    u3_mpc = mpc_signal(t_mpc, dt_mpc, p_mpc[3,:],iter)
    
    u_in = [u1_oc + u1_mpc, u2_oc + u2_mpc, u3_oc + u3_mpc]
    diffusion!(dθ, θ, plate, property, boundary, actuation, u_in)
end


# Temperature measurement 
function measure_temperature(temperature_data, iosetup, index, pos_sensor)
    cellidx, chars = getSensing(iosetup, index, pos_sensor)
    temps = temperature_data[cellidx,:]
    measured_data = measure(temps, chars)
    return measured_data
end


# Smooth step function
function psi_smooth_step(t)
    tshift = 150            # Time shift until cosine is active 
    η = Tf/(Tf - 2*tshift) # Steepness

    if t<Tf*(η-1)/(2*η)
        return 0;
    elseif t>Tf*(η+1)/(2*η)
        return 1;
    else
        return (1-cospi(η*(t/Tf - (η-1)/(2*η))))/2;
    end 
end

Δr = 100;
ref(t) = θ₀ + Δr*psi_smooth_step(t)
#ref_data = ref.(sol.t)

# tshift = 150,  Δr=200, Tf=600
p_oc = [
    11.299872480445739
    1.9963301625903012
    5.800154384164705
   11.298653378705675
    2.0033776047651353
    5.872380732962788
   11.299876633361908
    1.9963310177077667
    5.8001824394982195]

 

p_mpc=[
    9.7847   7.71691  8.62945  8.04458  8.47171  8.64711  5.73158  8.59798  8.62338  7.11493  8.57444  7.77341  7.81869  8.79168  7.88855  8.28936  7.741    6.23763  8.83499  7.37992
 9.652    7.45342  7.36406  7.33047  7.65228  8.21016  5.70294  8.24737  8.15391  7.13823  7.95053  7.38775  7.54748  8.15712  7.62292  8.01645  7.35997  6.10175  8.32949  7.05656
 9.78428  7.71814  8.63139  8.04023  8.47282  8.64792  5.73176  8.59671  8.62527  7.10802  8.57202  7.78115  7.81995  8.78496  7.90849  8.28854  7.73327  6.23605  8.83707  7.38046
]

using OrdinaryDiffEq
θinit = θ₀*ones(Ntotal)
Toc = 600.0;
tspan = (0.0, Toc)

t_samp = 1.0
alg = KenCarp5()
heat_eq!(dθ, θ, p_mpc, t) = heat_conduction!(dθ, θ, p_mpc, t, 1; p_oc = p_oc)


T_offset = Toc;
Nsteps = 20; # Total MPC steps
dt_mpc = 20.; # MPC sampling time
Nmpc = 3 # MPC prediction horizon 
prob = ODEProblem(heat_eq!,θinit,tspan)
sol_oc = solve(prob,alg,p=zeros(3,1), saveat=t_samp)

Tf = Toc+Nsteps*dt_mpc
θinit_mpc = sol_oc[:,end];
tspan_mpc = (Toc, Tf)
prob_mpc = ODEProblem(heat_eq!,θinit_mpc,tspan_mpc)
sol_mpc = solve(prob_mpc,alg,p=exp.(p_mpc), saveat=t_samp)

nt_oc = length(sol_oc.t);
nt_mpc = length(sol_mpc.t);
tgrid = vcat(sol_oc.t, sol_mpc.t[2:end])
num_tsteps = length(tgrid)



    # Temperature measurement 
function measure_temperature(temperature_data, iosetup, index, pos_sensor)
    cellidx, chars = getSensing(iosetup, index, pos_sensor)
    temps = temperature_data[cellidx,:]
    measured_data = measure(temps, chars)
    return measured_data
end

temp_data = zeros(num_tsteps, num_actuators)

measure_temperature(sol_oc, sensing, 1, :north)
temp_data[1:nt_oc,1] = measure_temperature(sol_oc, sensing, 1, :north)
temp_data[1:nt_oc,2] = measure_temperature(sol_oc, sensing, 2, :north)
temp_data[1:nt_oc,3] = measure_temperature(sol_oc, sensing, 3, :north)

temp_data[nt_oc+1:end,1] = measure_temperature(sol_mpc[:,2:end], sensing, 1, :north)
temp_data[nt_oc+1:end,2] = measure_temperature(sol_mpc[:,2:end], sensing, 2, :north)
temp_data[nt_oc+1:end,3] = measure_temperature(sol_mpc[:,2:end], sensing, 3, :north)




# Smooth step function
# Time shift until cosine is active 
function psi_smooth_step(t;T = 1, tshift=50)
    η = T/(T - 2*tshift) # Steepness

    if t<T*(η-1)/(2*η)
        return 0;
    elseif t>T*(η+1)/(2*η)
        return 1;
    else
        return (1-cospi(η*(t/T - (η-1)/(2*η))))/2;
    end 
end

η = Toc/(Toc - 2*150) # Steepness

Δr = 200;
ref_cos(t) = θ₀ + Δr*psi_smooth_step(t;T=Toc, tshift=150)
ref_data = ref_cos.(tgrid)

error_data = zeros(num_tsteps, num_actuators)
error_data[:,1] = ref_data - temp_data[:,1]
error_data[:,2] = ref_data - temp_data[:,2]
error_data[:,3] = ref_data - temp_data[:,3]

t_ref = 0 : 50 : tgrid[end]



xs = plate.sampling[1]
xgrid = xs : xs : Nx*xs

temp_data_complete = zeros(num_tsteps, Ntotal)
temp_data_complete[1:nt_oc,:] = copy(hcat(sol_oc.u[:]...)')
temp_data_complete[nt_oc+1:end,:] = copy(hcat(sol_mpc.u[2:end]...)')

temp_data_complete
temp_data_complete[:,(Ny-1)*Nx+1:Ny*Nx]
using CairoMakie

begin
    fig = Figure(size=(800,600),fontsize=20)
    ax = Axis3(fig[1,1], azimuth = 5pi/4, 
                xlabel = "Time t in [s]", ylabel = "Position x in [m]", zlabel = "Temperature in [K]", 
                xlabelsize = 24, ylabelsize = 24, zlabelsize = 24,)

    surface!(ax, tgrid[1:20:end], xgrid, temp_data_complete[1:20:end,(Ny-1)*Nx+1:Ny*Nx], colormap = :plasma)            
    fig
    save("results/figures/"*"temp_north_3d.pdf", fig,pt_per_unit = 1)    
end


u_in_data = zeros(num_tsteps, num_actuators)

u_in_data[1:nt_oc,1] = input_signal_oc.(tgrid[1:nt_oc]/Toc,p_oc[1:3]...)
u_in_data[1:nt_oc,2] = input_signal_oc.(tgrid[1:nt_oc]/Toc,p_oc[4:6]...)
u_in_data[1:nt_oc,3] = input_signal_oc.(tgrid[1:nt_oc]/Toc,p_oc[7:9]...)

mpc_time = tgrid[nt_oc+1:end] .- T_offset
blubb_mpc = mpc_signal.(mpc_time', dt_mpc, exp.(p_mpc[1,:]),0)

mpc_signal.(mpc_time, dt_mpc, exp.(p_mpc[1,1]),0)
u_in_data[nt_oc+1:end,1] = mapreduce(i->mpc_signal.(mpc_time, dt_mpc, exp.(p_mpc[1,i+1]),i),+,0:size(p_mpc)[2]-1)
u_in_data[nt_oc+1:end,2] = mapreduce(i->mpc_signal.(mpc_time, dt_mpc, exp.(p_mpc[2,i+1]),i),+,0:size(p_mpc)[2]-1)
u_in_data[nt_oc+1:end,3] = mapreduce(i->mpc_signal.(mpc_time, dt_mpc, exp.(p_mpc[3,i+1]),i),+,0:size(p_mpc)[2]-1)



begin
    fig1 = Figure(size=(800,600),fontsize=20)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "Input Signal", ylabelsize = 24,
        xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = 0 : 100 : tgrid[end];    
    #ax1.yticks = -5 : 0.5 : 3;
    lines!(tgrid, u_in_data[:,1]; linestyle = :dot,     linewidth = 4, label = L"Input 1$")
    lines!(tgrid, u_in_data[:,2]; linestyle = :dash,    linewidth = 4, label = L"Input 2$")
    lines!(tgrid, u_in_data[:,3]; linestyle = :dashdot, linewidth = 4, label = L"Input 3$")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1));

    t0_ax2 = 600;
    x_off = 0;
    y_off = 0;

    ax2 = Axis(fig1, bbox=BBox(425, 753, 317, 538), ylabelsize = 24, title=L"$\times 10^3$                                       ")
    #ax2 = Axis(fig1, bbox=BBox(435, 750, 231, 437), ylabelsize = 24, title=L"$\times 10^3$                                       ")
    #ax2.xticks =  t0_ax2: 50 : Tf;
    #ax2.yticks = 496 : 2 : 504;
    lines!(ax2, tgrid[t0_ax2:end],u_in_data[t0_ax2:end,1] / 1e3; linestyle = :dot,     linewidth = 3, color=Makie.wong_colors()[1])
    lines!(ax2, tgrid[t0_ax2:end],u_in_data[t0_ax2:end,2] / 1e3; linestyle = :dash,    linewidth = 3, color=Makie.wong_colors()[2])
    lines!(ax2, tgrid[t0_ax2:end],u_in_data[t0_ax2:end,3] / 1e3; linestyle = :dashdot, linewidth = 3, color=Makie.wong_colors()[3])
    translate!(ax2.scene, 0, 0, 0);

    fig1
    save("results/figures/"*"input_signal.pdf", fig1, pt_per_unit = 1)    
  end


begin
    fig1 = Figure(size=(800,600),fontsize=20)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "Temperature in [K]", ylabelsize = 24,
        xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = 0 : 100 : Tf;    
    ax1.yticks = θ₀ : 20 : θ₀+Δr;
    lines!(tgrid, temp_data[:,1]; linestyle = :dot,     linewidth = 4, label = "Sensor 1")
    lines!(tgrid, temp_data[:,2]; linestyle = :dash,    linewidth = 4, label = "Sensor 2")
    lines!(tgrid, temp_data[:,3]; linestyle = :dashdot, linewidth = 4, label = "Sensor 3")
    scatter!(t_ref, ref_data[1:50:end]; markersize = 15, marker = :diamond, color=:black, label = "Reference")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1));
  
    t0_ax2 = 550;
    x_off = 0;
    y_off = 0;

    # ax2 = Axis(fig1, bbox=BBox(515, 750, 265, 440), ylabelsize = 24)
    ax2 = Axis(fig1, bbox=BBox(435, 750, 231, 437), ylabelsize = 24)
    #ax2.xticks =  t0_ax2: 50 : Tf;
    #ax2.yticks = 496 : 2 : 504;
    lines!(ax2, tgrid[t0_ax2:end],temp_data[t0_ax2:end,1]; linestyle = :dot,     linewidth = 5, color=Makie.wong_colors()[1])
    lines!(ax2, tgrid[t0_ax2:end],temp_data[t0_ax2:end,2]; linestyle = :dash,    linewidth = 5, color=Makie.wong_colors()[2])
    lines!(ax2, tgrid[t0_ax2:end],temp_data[t0_ax2:end,3]; linestyle = :dashdot, linewidth = 5, color=Makie.wong_colors()[3])
    scatter!(ax2, tgrid[t0_ax2:50:end], ref_data[t0_ax2:50:end]; markersize = 15, marker = :diamond, color=:black)
    translate!(ax2.scene, 0, 0, 0);
    fig1
    save("results/figures/"*"he_ref_tracking.pdf", fig1, pt_per_unit = 1)    
  end




begin
    fig1 = Figure(size=(800,600),fontsize=20)
    ax1 = Axis(fig1[1, 1], xlabel = "Time t in [s]", ylabel = "Temperature in [K]", ylabelsize = 24,
        xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
        xtickalign = 1., xticksize = 10, 
        xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
        yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
        ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    
    ax1.xticks = 0 : 100 : tgrid[end];    
    ax1.yticks = -5 : 0.5 : 3;
    lines!(tgrid, error_data[:,1]; linestyle = :dot,     linewidth = 4, label = L"Error $e_{1}$")
    lines!(tgrid, error_data[:,2]; linestyle = :dash,    linewidth = 4, label = L"Error $e_{2}$")
    lines!(tgrid, error_data[:,3]; linestyle = :dashdot, linewidth = 4, label = L"Error $e_{3}$")
    axislegend(; position = :lt, backgroundcolor = (:grey90, 0.1));
    fig1
    save("results/figures/"*"error_trajectory_cos_ref.pdf", fig1, pt_per_unit = 1)    
  end
  

