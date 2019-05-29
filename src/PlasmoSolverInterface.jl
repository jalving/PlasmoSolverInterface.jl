module PlasmoSolverInterface

using AlgebraicGraphs
using JuMP
using Libdl
using MPI
using Distributed

#import Plasmo: solve, AbstractPlasmoSolver

export PipsNlpSolver,pipsnlp_solve

#include("solve.jl")

include("PipsNlpInterface.jl")

using .PipsNlpInterface

include("solve.jl")

end # module
