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

con_data = get_constraint_data(m::JuMP.Model)

d = NLPEvaluator(m)

constraintbounds(con_data)
pips_jacobian_structure(d)
