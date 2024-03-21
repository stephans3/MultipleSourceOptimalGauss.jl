
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


function heat_conduction!(dθ, θ, param, t)
    time = t/Toc
    u1 = input_signal_oc(time,param[1:3]...)
    u2 = input_signal_oc(time,param[4:6]...)
    u3 = input_signal_oc(time,param[7:9]...)
    
    u_in = [u1, u2, u3]
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
function loss_output(p)
    sol_temp = solve(prob, alg, p=p, saveat = t_samp)
    
    if sol_temp.retcode != ReturnCode.Success
        return Inf
    end

    num_tsteps = length(sol_temp.t)
    temp_data = zeros(num_tsteps, 0)
    
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 1, :north))
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 2, :north))
    temp_data = hcat(temp_data, measure_temperature(sol_temp, sensing, 3, :north))

    err = sum((temp_data .- ref_data).^2, dims=2)/num_sensor
    loss = sum(err) / num_tsteps
    return loss
end

const loss_store = Float64[]
const loss_idx_l = 3
const loss_var = 0.001
callback = function (p, l)
    display(l)

    append!(loss_store,l)

    
    if length(loss_store) > loss_idx_l
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


θinit = θ₀*ones(Ntotal)
Toc = 600.0;
tspan = (0.0, Toc)

# Parameters from ff_control_energy_based.jl, Δr=200, Toc=600
p_energy = round.([11.49419489021336, 2.0,7.428803398450493], digits=3)
p0 = repeat(p_energy,3)
#p0 = repeat([11.5,2,7],3)
using OrdinaryDiffEq
t_samp = 2.0
alg = KenCarp5()
prob = ODEProblem(heat_conduction!,θinit,tspan,p0)
sol = solve(prob,alg,p=p0, saveat=t_samp)

Δr = 200;
ref(t) = θ₀ + Δr*psi_smooth_step(t,T=Toc,tshift=150)
ref_data = ref.(sol.t)

# Test cost function
pinit = copy(p0)
loss_output(pinit)

# Parameter Optimization
using Optimization, SciMLSensitivity 
using ForwardDiff
using OptimizationOptimJL

adtype = Optimization.AutoForwardDiff();
optf = Optimization.OptimizationFunction((x, p) -> loss_output(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit, lb=repeat([0.0, 1.0, 0.0],3), ub=repeat([20, Inf, Inf],3));
opt_pars = Optimization.solve(optprob, BFGS(), callback = callback, maxiters = 20); 

p_found = opt_pars.u

#= tshift = 150,  Δr=200, Toc=600
11.299872480445739
  1.9963301625903012
  5.800154384164705
 11.298653378705675
  2.0033776047651353
  5.872380732962788
 11.299876633361908
  1.9963310177077667
  5.8001824394982195
=#
