module PlasmoSolverInterface

using AlgebraicGraphs
using JuMP
using Libdl
using MPI
using Distributed

#import Plasmo: solve, AbstractPlasmoSolver

export PipsSolver,pipsnlp_solve

#include("solve.jl")

include("PipsNlpInterface.jl")

using .PipsNlpInterface

end # module
