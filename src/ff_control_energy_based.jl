# Feed-forward control
# Energy-based approach with implicit function

# System properties
θ₀ = 300.0  # Initial temperature
λ = 45.0    # Thermal conductivity: constant
ρ = 7800.0  # Mass density: constant
c = 400.0   # Specific heat capacity: constant
L = 0.3     # Length
W = 0.02    # Width

# Actuator characteristics
scale     = 1.0;
power     = 3;
curvature = 20.0;
num_act   = 3;
xc        = 0.05 # 1. central point
spat_char(x) = scale*exp(-(curvature*(x-xc))^(2*power))
Δr = 200; # Reference: increase of temperature 
T=600; # Final simulation time

xgrid = 0 : 0.001 : 0.1;
using Plots
scatter(xgrid, spat_char.(xgrid))

using FastGaussQuadrature
x,w = FastGaussQuadrature.gausslegendre(1000)
spat_char_int = xc*FastGaussQuadrature.dot(w,spat_char.(xc*x .+ xc))

E_in = (1/(num_act*spat_char_int))*c*ρ*L*W*Δr

using SpecialFunctions
using NLsolve


function input_signal_oc(time,p₁,p₂,p₃)
    return  exp(p₁- p₃^2 * (time - 1/p₂)^2)
end

u_in_energy(p₁,p₂,p₃) = exp(p₁)*T*sqrt(pi) * (erf(p₃-p₃/p₂) - erf(-p₃/p₂)) / (2 * p₃)

function nl_input!(F, x)
    q = 0.1;
    p₂ = 2;
    F[1] = input_signal_oc(0,x[1], p₂, x[2]) - q 
    F[2] = u_in_energy(x[1], p₂, x[2]) - E_in
end

sol = nlsolve(nl_input!, [10.0, 10.0])
p1,p3 = sol.zero

p3 = abs(p3)

# Found parameters:
pars = [p1,2,p3]

#= Tf=400
Δr=100;
pars = 
[11.195571832790092
 2.0
 7.3479675899623125]
=#

#= Tf=400
Δr=200;
pars = 
[13.134859275244763
  2.0
 25.548132508960773]
=#

#= Tf=600
Δr=200;
pars = 
[11.49419489021336
  2.0
  7.428803398450493]
=#
