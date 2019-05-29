mutable struct PipsNlpSolver <: AbstractGraphSolver
    options::Dict{Symbol,Any}
    manager::MPI.MPIManager
    status::Symbol
end

function PipsNlpSolver(;n_workers = 1)
    solver = PipsNlpSolver(
    Dict(:n_workers => n_workers),
    #Dict(:master_partition => master_partition,:sub_partitions => partitions),
    MPI.MPIManager(np = n_workers),
    :not_executed)
end

function JuMP.optimize!(graph::ModelGraph,solver::PipsNlpSolver)

    manager = solver.manager
    if length(manager.mpi2j) == 0
        addprocs(manager)
    end

    println("Preparing PIPS MPI environment")
    eval(quote @everywhere using Pkg end)
    #TODO: Better way to load environment.  Use project flags?
    eval(quote @everywhere Pkg.activate("/home/jordan/.julia/dev/PlasmoSolverInterface") end)
    eval(quote @everywhere using PlasmoSolverInterface end)


    println("Finished preparing environment")

    println("Sending model to workers")
    send_pips_data(manager,graph)

    println("Solving with PIPS-NLP")
    #NOTE Need to get status = pipsnlp_solve

    MPI.@mpi_do manager pipsnlp_solve(graph)

    #Get solution
    rank_zero = manager.mpi2j[0]
    sol = fetch(@spawnat(rank_zero, getfield(Main, :graph)))

    #Update the graph on the julia process if we used a PipsTree
    #setsolution(sol,pips_graph)

    #Now move the pips_graph solution to the original model graph
    #setsolution(pips_graph,graph)

    return nothing  #TODO retrieve solve status
end

function send_pips_data(manager::MPI.MPIManager,graph::ModelGraph)
    julia_workers = collect(values(manager.mpi2j))
    r = RemoteChannel(1)
    @spawnat(1, put!(r, [graph]))
    @sync for to in julia_workers
        @spawnat(to, Core.eval(Main, Expr(:(=), :graph, fetch(r)[1])))
    end
end

# #load PIPS-NLP if the library can be found
# function load_pips()
#     if  !isempty(Libdl.find_library("libparpipsnlp"))
#         #include("solver_interfaces/plasmoPipsNlpInterface3.jl")
#         #eval(quote using .PlasmoPipsNlpInterface3 end)
#         eval(macroexpand(quote @everywhere using .PlasmoPipsNlpInterface3 end))
#     else
#         pipsnlp_solve(Any...) = throw(error("Could not find a PIPS-NLP installation"))
#     end
# end

#load DSP if the library can be found
# function load_dsp()
#     if !isempty(Libdl.find_library("libDsp"))
#         include("solver_interfaces/plasmoDspInterface.jl")
#         using .PlasmoDspInterface
#     else
#         dsp_solve(Any...) = throw(error("Could not find a DSP installation"))
#     end
# end
