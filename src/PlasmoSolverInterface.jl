module PlasmoSolverInterface

using Plasmo
using Libdl
using MPI
using Distributed

import Plasmo: solve, AbstractPlasmoSolver

export PipsSolver,pipsnlp_solve,dsp_solve

include("wrapped_solvers.jl")

include("plasmoPipsNlpInterface.jl")

include("DspCInterface.jl")

include("plasmoDspInterface.jl")

using .PlasmoPipsNlpInterface

using .PlasmoDspInterface

end # module
