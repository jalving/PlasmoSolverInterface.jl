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
#demand = rand(Ns)*10
#NOTE: Too many significant figures leads to Restoration_needed in Pips-nlp
demand = [0.46,7.67,3.70,0.07,0.58,6.11,5.36,5.78,1.22,0.81,1.36,4.70,4.81,0.29,5.57]
graph = ModelGraph()

#Create the master model NOTE: The PIPS-Interface isn't using a Master model right now, it is empty by default.  I am going to make an example that uses @graphvariable and @graphobjective
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

ipopt_optimizer = with_optimizer(Ipopt.Optimizer)
optimize!(graph,ipopt_optimizer)
m = graph.jump_model
ipopt_solution = JuMP.value.(JuMP.all_variables(m))

pipsnlp_solve(graph)
pips_solution  = vcat([getmodel(node).ext[:colVal] for node in getnodes(graph)]...)

println(ipopt_solution - pips_solution) #NOTE: Should be almost zero

#MPI.Finalize()
