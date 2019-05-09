using Pkg
Pkg.activate("../")

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Revise

include("helpers.jl")

m = Model()
@variable(m,x[1:3] >= 0)
@constraint(m,x[1] + x[2] <= 2)
@constraint(m, 0 <= x[2] + x[3] <= 5)
@constraint(m,x[1] + x[3]^2 >= 4)
@NLconstraint(m,x[3]^3 >= 4)
@objective(m,Min,x[1]^2 + x[1]*x[2] + x[3] + 2)

con_data = get_constraint_data(m::JuMP.Model)
m.ext[:constraint_data] = con_data

d = NLPEvaluator(m)

bounds = constraintbounds(con_data)
jac_structure = pips_jacobian_structure(d)
hess_structure = pips_hessian_lagrangian_structure(d)

x = Float64[1,2,3]

obj = pips_eval_objective(d,x)

grad = zeros(length(x))
pips_eval_objective_gradient(d,grad,x)

g_constraint = zeros(numconstraints(m))
pips_eval_constraint(d,g_constraint,x)

jac_values = Array{Float64}(undef,length(jac_structure))
pips_eval_constraint_jacobian(d,jac_values,x)

hess_values = Array{Float64}(undef,length(hess_structure))
pips_eval_hessian_lagrangian(d,hess_values,x,1.0,ones(length(hess_values)))

var_lower = variablelowerbounds(m)
var_upper = variableupperbounds(m)
