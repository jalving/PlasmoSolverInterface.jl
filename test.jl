using Pkg
Pkg.activate("./")

using MPI
#MPI.Init()
using Revise
using PlasmoSolverInterface
using AlgebraicGraphs
using JuMP
using Ipopt



function simple_electricity_model(demand)
    m = Model()
    #amount of electricity produced
    @variable(m, 0<=prod<=10, start=5)
    #amount of electricity purchased or sold
    @variable(m, input, start = 2)
    #amount of gas purchased
    @variable(m, gas_purchased, start = 2)
    @constraint(m, gas_purchased >= prod)
    @constraint(m, prod + input == demand)
    return m
end

Ns = 15
demand = rand(Ns)*10
graph = ModelGraph()

#Create the master model
master = Model()
@variable(master,0<=gas_purchased<=8, start = 2)
@objective(master,Min,gas_purchased)

#Add the master model to the graph
master_node = add_node!(graph,master)
#scen_models = Array{JuMP.Model}(undef,Ns)
scen_models = JuMP.Model[]
for j in 1:Ns
    scenm = simple_electricity_model(demand[j])
    node = add_node!(graph,scenm)

    #connect children and parent variables
    @linkconstraint(graph, master[:gas_purchased] == scenm[:gas_purchased])

    #Create child objective
    @objective(scenm,Min,1/Ns*(scenm[:prod] + 3*scenm[:input]))
    push!(scen_models,scenm)
end

#create a link constraint between the subproblems
@linkconstraint(graph, (1/Ns)*sum(scen_models[s][:prod] for s in 1:Ns) == 8)

pipsnlp_solve(graph)

MPI.Finalize()
#pipsnlp_solve(graph,solver)
