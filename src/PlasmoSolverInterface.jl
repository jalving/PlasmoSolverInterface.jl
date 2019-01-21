module PlasmoSolverInterface

using Plasmo
using Libdl
using MPI
using Distributed

import Plasmo: solve, AbstractPlasmoSolver

export PipsSolver,pipsnlp_solve

include("wrapped_solvers.jl")

include("plasmoPipsNlpInterface.jl")

using .PlasmoPipsNlpInterface

end # module
