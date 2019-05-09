using JuMP
using MathProgBase
const MPB = MathProgBase
using Revise

m = Model()
@variable(m,x[1:3] >= 0)
@constraint(m,x[1] + x[2] <= 2)
@constraint(m, 0 <= x[2] + x[3] <= 5)
@constraint(m,x[1] + x[3]^2 >= 4)
@NLconstraint(m,x[3]^3 >= 4)
@objective(m,Min,x[1]^2 + x[1]*x[2] + x[3] + 2)

d = JuMP.NLPEvaluator(m)
MPB.initialize(d,[:Jac,:Hess,:Grad])

bounds = JuMP.constraintbounds(m)
jac_structure = MPB.jac_structure(d)    #NOTE: duplicate entry here, which I'm not getting from my implementation
hess_structure = MPB.hesslag_structure(d)

x = Float64[1,2,3]

obj = MPB.eval_f(d,x)

grad = zeros(length(x))
MPB.eval_grad_f(d,grad,x)
numconstraints = MPB.SolverInterface.numconstr(m)
g_constraint = zeros(numconstraints)
MPB.eval_g(d,g_constraint,x)

jac_values = Array{Float64}(undef,length(jac_structure[1]))
MPB.eval_jac_g(d,jac_values,x)


hess_values = Array{Float64}(undef,length(hess_structure[1]))
MPB.eval_hesslag(d,hess_values,x,1.0,ones(length(hess_values)))
