
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
setIOSetup!(actuation, plate, num_actuators, config_a,  :south)

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

#=
tgrid = 0:0.01:1
my_step(t) = step_gauss(t,1,1)
using Plots
plot(my_step.(tgrid))
=#

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


# Cost function
function my_loss(p,prob_ode)
    p = exp.(p)
    sol_temp = solve(prob_ode, alg, p=p, saveat = t_samp)
    
    if sol_temp.retcode != ReturnCode.Success
        return Inf
    end

    num_tsteps = length(sol_temp.t)
    temp_data = zeros(num_tsteps, 0)
    
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 1, :north))
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 2, :north))
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 3, :north))
    ref_data = ref_cos.(sol_temp.t)
    err = sum((temp_data .- ref_data).^2, dims=2)/num_sensor
    loss = sum(err) / num_tsteps
    return loss
end



const loss_store = Float64[]
const max_loss_store = 20;
const loss_idx_l = 3
const loss_var = 0.1
callback = function (p, l)
    display(l)

    append!(loss_store,l)

    if length(loss_store) > max_loss_store
        return true
    elseif length(loss_store) > loss_idx_l
        a = loss_store[end-2:end]
        std_var = sqrt(abs(sum(a.^2)/3 - (sum(a)/3)^2))
        if std_var < loss_var
            return true
        end
    end

    if l < 1 
        return true
    end
    return false
end


# Parameters from opt_con_ref_tracking_cos.jl
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

Δr = 200;
Toc = 600.0; # Simulation Time: Optimization-based feed-forward control 
ref_cos(t) = θ₀ + Δr*psi_smooth_step(t;T=Toc, tshift=150)


using OrdinaryDiffEq
θinit = θ₀*ones(Ntotal)
tspan = (0.0, Toc)
t_samp = 1.0
alg = KenCarp5()
heat_eq!(dθ, θ, p_mpc, t) = heat_conduction!(dθ, θ, p_mpc, t, 1; p_oc = p_oc)


T_offset = Toc;
Nsteps = 20; # Total MPC steps
dt_mpc = 20.; # MPC sampling time
Nmpc = 3 # MPC prediction horizon 
prob = ODEProblem(heat_eq!,θinit,tspan)
sol_oc = solve(prob,alg,p=zeros(num_actuators,1), saveat=t_samp)


Tf = Toc+Nsteps*dt_mpc
θinit_mpc = sol_oc[:,end];


next_param = 5ones(num_actuators,Nmpc)
lossArray = zeros(Nsteps+1)
paramArray = zeros(num_actuators, Nsteps);
paramArray[:,1] = next_param[:,1];

θinit = sol_oc[:,end]

using Optimization, OptimizationOptimJL, SciMLSensitivity, ForwardDiff
adtype = Optimization.AutoForwardDiff();


for i = 0 : Nsteps-1
    tspan_opt = (T_offset + i*dt_mpc, T_offset + (i+Nmpc)*dt_mpc)
    heat_eq_mpc!(dx,x,p,t) = heat_conduction!(dx,x,p,t,i; p_oc=p_oc)
    prob_mpc = ODEProblem(heat_eq_mpc!, θinit, tspan_opt)
    my_loss2(x) = my_loss(x,prob_mpc) 

    # my_loss2(next_param)

    optf = Optimization.OptimizationFunction((x, p) -> my_loss2(x), adtype);
    lower_bounds = -Inf*ones(3,Nmpc)
    upper_bounds = 20ones(3,Nmpc)
    optprob = Optimization.OptimizationProblem(optf, next_param, lb=lower_bounds, ub=upper_bounds);
    opt_res = Optimization.solve(optprob, BFGS(), callback=callback, maxiters = 10);
    empty!(loss_store);

    paramArray[:,i+1] = opt_res.u[:,1]
    next_param = hcat(opt_res.u[:,2:end], 5ones(num_actuators))
    prob1 = remake(prob_mpc; tspan=(T_offset + i*dt_mpc, T_offset + (i+1)*dt_mpc))
        
    pars_new = exp.(opt_res.u[:,1])
    sol123 = solve(prob1,alg, p=pars_new, saveat=t_samp)
    θinit = sol123[:,end]
    println("Iteration i=",i)
end



paramArray

using DelimitedFiles;
path2folder = "results/data/"
filename = "mpc_param_data_dr_200_Toc_600_Tmpc_400.txt"

path2file = path2folder * filename

open(path2file, "w") do io
    writedlm(io, paramArray, ',')
end;



