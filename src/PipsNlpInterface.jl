module PipsNlpInterface

#Import Packages
using SparseArrays
using LinearAlgebra
import MPI

import JuMP  #Model Interface
import MathOptInterface #Lowlevel Interface
const MOI = MathOptInterface

using AlgebraicGraphs  #ModelGraph Interface


include("PipsNlpCInterface.jl")
using .PipsNlpSolver

include("helpers.jl")
export pipsnlp_solve

mutable struct ModelData
    d::Union{JuMP.NLPEvaluator,Nothing}               #NLP evaluator
    n::Int
    m::Int
    local_m::Int
    jacnnz::Int                     #Jacobian nonzeros
    hessnnz::Int                    #Hessian nonzeros
    firstIeq::Vector{Int}           #row index of equality constraint in 1st stage
    firstJeq::Vector{Int}           #column index of equality constraint in 1st stage
    firstVeq::Vector{Float64}       #coefficient of variable in equality constraint in 1st stage
    secondIeq::Vector{Int}          #row index of equality constraint in 2nd stage
    secondJeq::Vector{Int}          #column index of equality constraint in 2nd stage
    secondVeq::Vector{Float64}      #coefficient of variable in equality constraint in 2nd stage
    firstIineq::Vector{Int}
    firstJineq::Vector{Int}
    firstVineq::Vector{Float64}
    secondIineq::Vector{Int}
    secondJineq::Vector{Int}
    secondVineq::Vector{Float64}
    num_eqconnect::Int
    num_ineqconnect::Int
    eqconnect_lb::Vector{Float64}
    eqconnect_ub::Vector{Float64}
    ineqconnect_lb::Vector{Float64}
    ineqconnect_ub::Vector{Float64}
    eq_idx::Vector{Int}
    ineq_idx::Vector{Int}
    firstJeqmat
    secondJeqmat
    firstJineqmat
    secondJineqmat
    linkIeq::Vector{Int}
    linkJeq::Vector{Int}
    linkVeq::Vector{Float64}
    linkIineq::Vector{Int}
    linkJineq::Vector{Int}
    linkVineq::Vector{Float64}
    x_sol::Vector{Float64}
    coreid::Int
    loaded::Bool
    local_unsym_hessnnz::Int   #number of nonzero values in the unsymmetric hessian (half the hessian)
end
ModelData() = ModelData(nothing,0,0,0,0,0,Int[],Int[], Float64[], Int[], Int[], Float64[],Int[],Int[],Float64[], Int[], Int[], Float64[], 0, 0, Float64[],Float64[],Float64[],Float64[], Int[], Int[],nothing, nothing, nothing, nothing, Int[],Int[], Float64[], Int[], Int[], Float64[],Float64[], 0, false, 0)

#Helper function
function getData(m::JuMP.Model)
    if haskey(m.ext, :Data)
        return m.ext[:Data]
    else
        error("This functionality is only available to model with extension data")
    end
end

# function pipsnlp_solve(graph::PlasmoModelGraph.AbstractModelGraph,master_index::Int64,children_indices::Vector{Int64})
function pipsnlp_solve(graph::ModelGraph,master_index::Int64)
end

function pipsnlp_solve(graph::ModelGraph) #Assume graph variables and constraints are first stage
    #NOTE: Assume no master model if not given a master index.  We will create a dummy model
    println("Solving with PIPS-NLP")
    master_node = ModelNode()
    submodels = [getmodel(node) for node in getnodes(graph)]
    scen = length(submodels)
    #############################
    master = getmodel(master_node)
    modelList = [master; submodels]  #NOTE: this function creates an empty master problem

    #Add ModelData to each model
    for (idx,node) in enumerate(modelList)
        node.ext[:Data] = ModelData()
    end

    # master (link) constraint upper and lower bounds
    # master_linear_lb = []
    # master_linear_ub = []


    #TODO: get_all_linkconstraints.  We don't support subgraphs for this solver.
    #TODO: Check for linkconstraints between a master node and children nodes
    linkconstraints = getlinkconstraints(graph) #get all of the link constraints in the graph

    #get arrays of lower and upper bounds for each link constraint
    # for con in linkconstraints
    #     push!(master_linear_lb,con.lb)
    #     push!(master_linear_ub,con.ub)
    # end

    #Store data about links
    #NOTE: num_eqconnect and num_ineqconnect are zero since there is technically no master node here.
    nlinkeq = 0                 #Link constraint equalities
    nlinkineq = 0               #Link constraint inequalities
    eqlink_lb = Float64[]       #
    eqlink_ub = Float64[]
    ineqlink_lb = Float64[]
    ineqlink_ub = Float64[]

    #go through the link constraints
    for link in linkconstraints
        # coeffs = link.func.terms.vals
        # vars = link.func.terms.keys
	    #LINKCONSTRAINTS BETWEEN SUBPROBLEMS
        if isa(link.set,MOI.EqualTo)  #EQUALITY CONSTRAINTS
            nlinkeq = nlinkeq + 1
            push!(eqlink_lb, link.set.value)#master_linear_lb[c])
            push!(eqlink_ub, link.set.value)#master_linear_ub[c])
            row = nlinkeq
            for (var,coeff) in link.func.terms
                node = var.model
                local_data = getData(node)
                linkIeq = local_data.linkIeq
                linkJeq = local_data.linkJeq
                linkVeq = local_data.linkVeq
                push!(linkIeq, row)             #the variable row
                push!(linkJeq, var.index.value)    #the variable column
                push!(linkVeq, coeff)             #the coefficient
            end
        else #LOOK FOR INEQUALITY CONSTRAINTS
            @assert typeof(link.set) in [MOI.Interval{Float64},MOI.LessThan{Float64},MOI.GreaterThan{Float64}]
            if isa(link.set,MOI.LessThan)
                nlinkineq = nlinkineq + 1
                push!(ineqlink_lb, -Inf)
                push!(ineqlink_ub, link.set.upper)
                row = nlinkineq
            elseif isa(link.set,MOI.GreaterThan)
                nlinkineq = nlinkineq + 1
                push!(ineqlink_lb, link.set.lower)
                push!(ineqlink_ub, Inf)
                row = nlinkineq
            elseif isa(link.set,MOI.Interval)
                nlinkineq = nlinkineq + 1
                push!(ineqlink_lb, link.set.lower)
                push!(ineqlink_ub, link.set.upper)
                row = nlinkineq
            end
            #Populate Connection Matrix
            for (var,coeff) in link.func.terms
                node = var.model
                local_data = getData(node)
                linkIineq = local_data.linkIineq
                linkJineq = local_data.linkJineq
                linkVineq = local_data.linkVineq
                push!(linkIineq, row)
                push!(linkJineq, var.index.value)
                push!(linkVineq, coeff)
            end
        end
    end
    master_data = getData(master)

    if haskey(master.ext, :nlinkeq)
        nlinkeq =  master.ext[:nlinkeq]
    end
    if haskey(master.ext, :nlinkineq)
        nlinkineq =  master.ext[:nlinkineq]
    end
    if haskey(master.ext, :eqlink_lb)
        eqlink_lb = copy(master.ext[:eqlink_lb])
        eqlink_ub = copy(master.ext[:eqlink_lb])
    end
    if haskey(master.ext, :eqlink_ub)
        eqlink_lb = copy(master.ext[:eqlink_ub])
        eqlink_ub = copy(master.ext[:eqlink_ub])
    end
    if haskey(master.ext, :ineqlink_lb)
        ineqlink_lb = copy(master.ext[:ineqlink_lb])
    end
    if haskey(master.ext, :ineqlink_ub)
        ineqlink_ub = copy(master.ext[:ineqlink_ub])
    end

    for (idx,node) in enumerate(modelList)
        local_data = getData(node)
        if haskey(node.ext, :linkIineq)
            local_data.linkIineq =  copy(node.ext[:linkIineq])
            node.ext[:linkIineq] = nothing
        end
        if haskey(node.ext, :linkJineq)
            local_data.linkJineq =  copy(node.ext[:linkJineq])
            node.ext[:linkJineq] = nothing
        end
        if haskey(node.ext, :linkVineq)
            local_data.linkVineq =  copy(node.ext[:linkVineq])
            node.ext[:linkVineq] = nothing
        end
        if haskey(node.ext, :linkIeq)
            local_data.linkIeq =  copy(node.ext[:linkIeq])
            node.ext[:linkIeq] = nothing
        end
        if haskey(node.ext, :linkJeq)
            local_data.linkJeq =  copy(node.ext[:linkJeq])
            node.ext[:linkJeq] = nothing
        end
        if haskey(node.ext, :linkVeq)
            local_data.linkVeq =  copy(node.ext[:linkVeq])
            node.ext[:linkVeq] = nothing
        end

        #Add constraint data for each node to help with function evaluations
        constraint_data = get_constraint_data(node)
        node.ext[:constraint_data] = constraint_data
    end


    #TODO: Test this works
    function str_init_x0(nodeid, x0)
        node = modelList[nodeid+1]
        jump_initval = copy(JuMP.start_value.(JuMP.all_variables(node)))
        local_initval = Vector{Float64}(undef,JuMP.num_variables(node))


        #TODO: Better initialization.  Picking variable values in bounds does not guarantee the initial point is feasible.
        nothing_values = isa.(jump_initval,Nothing)
        float_values = .!(nothing_values)
        local_initval[nothing_values] .= 1  #set to 1 by default
        local_initval[float_values] .= jump_initval[float_values]
        #local_initval = min.(max.(node.colLower,local_initval),node.colUpper)
        original_copy(local_initval,x0)
    end


    #TODO Update to JuMP 0.19 and MOI.
    #TODO Replace JuMP.constraintbounds, Replace all of the MOI functions with helpers
    function str_prob_info(nodeid,flag, mode,col_lb,col_ub,row_lb,row_ub)
        if flag != 1
            node = modelList[nodeid+1]
            local_data = getData(node)
            #Do a warm start on the node
            if !(local_data.loaded)
                local_data.loaded = true
                if (nodeid > 0)
                    if haskey(node.ext, :warmStart)
    	                if node.ext[:warmStart] == true
              	      	   solve(node)
                   	   	end
                    end
            	end

    			#nlp_lb, nlp_ub = JuMP.constraintbounds(node)  #This is every constraint in the model
                nlp_lb, nlp_ub = constraintbounds(node.ext[:constraint_data])       #This is every constraint in the model
         		local_data.local_m  = length(nlp_lb)          #number of local constraints (rows)

    			newRowId = Array{Int}(undef,local_data.local_m)
    			eqId = 1
    			ineqId = 1

                #Go through all constraints, check if they are equality of inequality
                for c in 1:local_data.local_m
                    if nlp_lb[c] == nlp_ub[c]
                        push!(local_data.eq_idx, c)
                        newRowId[c] = eqId
                        eqId +=  1
                    else
        				push!(local_data.ineq_idx, c)
        				newRowId[c] = ineqId
        				ineqId += 1
                    end
         		end

                #Local node data
                local_data.m  = local_data.local_m + local_data.num_eqconnect + local_data.num_ineqconnect
                #local_data.n = node.numCols
                local_data.n = JuMP.num_variables(node)
                local_data.x_sol = zeros(Float64,local_data.n)
                local_data.firstJeqmat = sparse(local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_data.num_eqconnect, master_data.n)
                local_data.secondJeqmat = sparse(local_data.secondIeq, local_data.secondJeq, local_data.secondVeq, local_data.num_eqconnect, local_data.n)
                local_data.firstJineqmat = sparse(local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_data.num_ineqconnect, master_data.n)
                local_data.secondJineqmat = sparse(local_data.secondIineq, local_data.secondJineq, local_data.secondVineq, local_data.num_ineqconnect, local_data.n)

                if node.nlp_data == nothing
                    JuMP._init_NLP(node)
                end
                local_data.d = JuMP.NLPEvaluator(node)
                MOI.initialize(local_data.d, [:Grad,:Jac, :Hess])
                #Ijac, Jjac = jac_structure(local_data.d)
                jac_structure = pips_jacobian_structure(local_data.d)
                Ijac = [jac_structure[i][1] for i = 1:length(jac_structure)]
                Jjac = [jac_structure[j][2] for j = 1:length(jac_structure)]
                Ijaceq = Int[]
                Jjaceq = Int[]
                Ijacineq = Int[]
                Jjacineq = Int[]
                jac_eq_index = Int[]
                jac_ineq_index = Int[]
                for i in 1:length(Ijac)
                    c = Ijac[i]
                    if nlp_lb[c] == nlp_ub[c]
                        modifiedrow = newRowId[c]
                        push!(Ijaceq, modifiedrow)
                        push!(Jjaceq, Jjac[i])
                        push!(jac_eq_index, i)
                    else
                        modifiedrow = newRowId[c]
                        push!(Ijacineq, modifiedrow)
                        push!(Jjacineq, Jjac[i])
                        push!(jac_ineq_index,i)
                    end
                end

        		node.ext[:Ijaceq] = Ijaceq
         		node.ext[:Jjaceq] = Jjaceq
                node.ext[:Ijacineq] = Ijacineq
         		node.ext[:Jjacineq] = Jjacineq
         		node.ext[:jac_eq_index] = jac_eq_index
         		node.ext[:jac_ineq_index] = jac_ineq_index
         		#Ihess, Jhess = hesslag_structure(local_data.d)
                hess_structure = pips_hessian_lagrangian_structure(local_data.d)
                Ihess = [hess_structure[i][1] for i = 1:length(hess_structure)]
                Jhess = [hess_structure[j][2] for j = 1:length(hess_structure)]

         		Hmap = Bool[]
         		node_Hrows = Int[]
         		node_Hcols = Int[]
         		for i in 1:length(Ihess)
            	    if Jhess[i] <= Ihess[i]
               	        push!(node_Hrows, Ihess[i])
                   		push!(node_Hcols, Jhess[i])
                   		push!(Hmap, true)
            	    else
                        push!(Hmap, false)
                    end
                end
         		val = ones(Float64,length(node_Hrows))
         		mat = sparse(node_Hrows,node_Hcols,val, local_data.n, local_data.n)
         		node.ext[:Hrows] = node_Hrows
         		node.ext[:Hcols] = node_Hcols
         		node.ext[:Hmap] = Hmap
         		local_hessnnz = length(mat.rowval)
         		local_data.local_unsym_hessnnz = length(Ihess)
         		local_data.hessnnz = local_hessnnz
		    end

	 	    if mode == :Values
               	#nlp_lb, nlp_ub = JuMP.constraintbounds(node)
                nlp_lb, nlp_ub = constraintbounds(node.ext[:constraint_data])
    			eq_lb=Float64[]
    			eq_ub=Float64[]
    			ineq_lb=Float64[]
    			ineq_ub=Float64[]
    			for i in 1: length(nlp_lb)
    			    if nlp_lb[i] == nlp_ub[i]
    			       push!(eq_lb, nlp_lb[i])
    			       push!(eq_ub, nlp_ub[i])
    			    else
    			       push!(ineq_lb, nlp_lb[i])
    			       push!(ineq_ub, nlp_ub[i])
    			    end
    			end

    			if nodeid !=  0 #0 is the master
    			   eq_lb = [eq_lb; local_data.eqconnect_lb]
    			   eq_ub = [eq_ub; local_data.eqconnect_ub]
    			   ineq_lb = [ineq_lb; local_data.ineqconnect_lb]
    			   ineq_ub = [ineq_ub; local_data.ineqconnect_ub]
    			end

    			original_copy([eq_lb;ineq_lb], row_lb)
    			original_copy([eq_ub;ineq_ub], row_ub)
                colLower = variablelowerbounds(node)
                colUpper = variableupperbounds(node)
    			# original_copy(node.colLower, col_lb)
    			# original_copy(node.colUpper, col_ub)
                original_copy(colLower, col_lb)
    			original_copy(colUpper, col_ub)
            end
		    return (local_data.n,local_data.m)
		else
		    if mode == :Values
                original_copy([eqlink_lb; ineqlink_lb], row_lb)
                original_copy([eqlink_ub; ineqlink_ub], row_ub)
            end
            return (0, nlinkeq + nlinkineq)
		end
    end

    #x0 is first stage variable values, x1 is local values
    function str_eval_f(nodeid,x0,x1)
        #println("eval_f")
    	node = modelList[nodeid+1] #Julia doesn't start index at 0
        local_data = getData(node)
        local_d = getData(node).d
        if nodeid ==  0
            local_x = x0
        else
            local_x = x1
        end
        #check objective sign
        #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
        local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
        #f = local_scl*eval_f(local_d,local_x)
        f = local_scl*pips_eval_objective(local_d,local_x)
        return f
    end

    function str_eval_g(nodeid,x0,x1,new_eq_g, new_inq_g)
        #println("eval_g")
        node = modelList[nodeid+1]
        local_data = getData(node)
        local_d = getData(node).d
        if nodeid ==  0
            local_x = x0
        else
            local_x = x1
        end
        local_g = Array{Float64}(undef,local_data.local_m)
        #eval_g(local_d, local_g, local_x)
        pips_eval_constraint(local_d, local_g, local_x)
        new_eq_g[1:end] = [local_g[local_data.eq_idx]; local_data.firstJeqmat*x0+local_data.secondJeqmat*x1]
        new_inq_g[1:end] = [local_g[local_data.ineq_idx]; local_data.firstJineqmat*x0+local_data.secondJineqmat*x1]
	    return Int32(1)
    end

    function str_write_solution(id::Integer, x::Vector{Float64}, y_eq::Vector{Float64}, y_ieq::Vector{Float64})
        node = modelList[id+1]
        local_data = getData(node)
        local_data.x_sol = copy(x)

        #NOTE: This won't work anymore
        if id == 0
            #Add objective value to node
            #node.objVal = str_eval_f(id, x, nothing)
            node.ext[:objective] = str_eval_f(id, x, nothing)
        else
            #node.objVal = str_eval_f(id, nothing, x)
            node.ext[:objective] = str_eval_f(id, nothing, x)
        end
        r = MPI.Comm_rank(comm)
        local_data.coreid = r
    end


    function str_eval_grad_f(rowid,colid,x0,x1,new_grad_f)
        #println("eval_grad_f")
        node = modelList[rowid+1]
        if rowid == colid
            local_data = getData(node)
            local_d = getData(node).d
            if colid ==  0
                local_x = x0
            else
                local_x = x1
            end
            local_grad_f = Array{Float64}(undef,local_data.n)
            pips_eval_objective_gradient(local_d, local_grad_f, local_x)
            #eval_grad_f(local_d, local_grad_f, local_x)
            #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
            local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
            #scale!(local_grad_f,local_scl)
            rmul!(local_grad_f,local_scl)
            original_copy(local_grad_f, new_grad_f)
        elseif colid == 0
            new_grad_f[1:end] .= 0
        else
            @assert(false)
        end
        return Int32(1)
    end

    #NOTE: Why does it subtract 1?
    function array_copy(src,dest)
        @assert(length(src)==length(dest))
        for i in 1:length(src)
            dest[i] = src[i]-1  #NOTE: Is this copying indices to a C vector?
        end
    end

    function original_copy(src,dest)
        @assert(length(src)==length(dest))
        for i in 1:length(src)
            dest[i] = src[i]
        end
    end

    function str_eval_jac_g(rowid,colid,flag, x0,x1,mode,e_rowidx,e_colptr,e_values,i_rowidx,i_colptr,i_values)
        #println("eval_jac_g")
        if flag != 1
            node = modelList[rowid+1]
            local_data = getData(node)
            local_m_eq = length(local_data.eq_idx)
            local_m_ineq = length(local_data.ineq_idx)
            if mode == :Structure
                if (rowid == colid)
                    Ieq=[node.ext[:Ijaceq];local_data.secondIeq .+ local_m_eq]
                    Jeq=[node.ext[:Jjaceq];local_data.secondJeq]
                    Veq= ones(Float64, length(Ieq))
                    Iineq=[node.ext[:Ijacineq];local_data.secondIineq .+ local_m_ineq]
                    Jineq=[node.ext[:Jjacineq];local_data.secondJineq]
                    Vineq=ones(Float64, length(Iineq))
                    eqmat = sparse(Ieq, Jeq, Veq, local_m_eq + local_data.num_eqconnect, local_data.n)
                    ineqmat = sparse(Iineq, Jineq, Vineq, local_m_ineq + local_data.num_ineqconnect, local_data.n)
                else
                    eqmat = sparse(local_m_eq .+ local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_m_eq .+ local_data.num_eqconnect, master_data.n)
                    ineqmat = sparse(local_m_ineq .+ local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_m_ineq .+ local_data.num_ineqconnect, master_data.n)
                end
                return(length(eqmat.rowval), length(ineqmat.rowval))
            else #mode = :Values
                if rowid == colid    #evaluate Jacobian for block
                    if colid ==  0   #if it's the root node
                        local_x = x0
                    else
                        local_x = x1
                    end
                    local_values = Array{Float64}(undef,length(node.ext[:Ijaceq])+length(node.ext[:Ijacineq]))
                    #eval_jac_g(local_data.d, local_values, local_x)
                    pips_eval_constraint_jacobian(local_data.d, local_values, local_x)
                    jac_eq_index = node.ext[:jac_eq_index]
                    jac_ineq_index = node.ext[:jac_ineq_index]
                    Ieq=[node.ext[:Ijaceq];local_data.secondIeq .+ local_m_eq]
                    Jeq=[node.ext[:Jjaceq];local_data.secondJeq]
                    Veq=[local_values[jac_eq_index];local_data.secondVeq]
                    Iineq=[node.ext[:Ijacineq];local_data.secondIineq .+ local_m_ineq]
                    Jineq=[node.ext[:Jjacineq];local_data.secondJineq]
                    Vineq=[local_values[jac_ineq_index];local_data.secondVineq]
                    eqmat = sparseKeepZero(Ieq, Jeq, Veq, local_m_eq + local_data.num_eqconnect, local_data.n)
                    ineqmat = sparseKeepZero(Iineq, Jineq, Vineq, local_m_ineq + local_data.num_ineqconnect, local_data.n)
                else #evaluate
                    eqmat = sparseKeepZero(local_m_eq .+ local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_m_eq .+ local_data.num_eqconnect, master_data.n)
                    ineqmat = sparseKeepZero(local_m_ineq .+ local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_m_ineq .+ local_data.num_ineqconnect, master_data.n)
                end
                if length(eqmat.nzval) > 0
                    array_copy(eqmat.rowval,e_rowidx)
                    array_copy(eqmat.colptr,e_colptr)
                    original_copy(eqmat.nzval,e_values)
                end
                if length(ineqmat.nzval) > 0
                    array_copy(ineqmat.rowval,i_rowidx)
                    array_copy(ineqmat.colptr,i_colptr)
                    original_copy(ineqmat.nzval, i_values)
                end
            end
        else
            node = modelList[rowid+1]
            local_data = getData(node)
            linkIeq = local_data.linkIeq
            linkJeq = local_data.linkJeq
            linkVeq = local_data.linkVeq
            linkIineq = local_data.linkIineq
            linkJineq = local_data.linkJineq
            linkVineq = local_data.linkVineq

            if mode == :Structure
                return(length(linkVeq), length(linkVineq))
            else
                eqmat = sparse(linkIeq, linkJeq, linkVeq, nlinkeq, local_data.n)
                ineqmat = sparse(linkIineq, linkJineq, linkVineq, nlinkineq, local_data.n)
                if length(eqmat.nzval) > 0
                    array_copy(eqmat.rowval,e_rowidx)
                    array_copy(eqmat.colptr,e_colptr)
                    original_copy(eqmat.nzval,e_values)
                end
                if length(ineqmat.nzval) > 0
                    array_copy(ineqmat.rowval,i_rowidx)
                    array_copy(ineqmat.colptr,i_colptr)
                    original_copy(ineqmat.nzval, i_values)
                end
            end
        end
        return Int32(1)
    end

    function str_eval_h(rowid,colid,x0,x1,obj_factor,lambda,mode,rowidx,colptr,values)
        #println("eval_h")
        node = modelList[colid+1]
        local_data = getData(node)
        if mode == :Structure
            if rowid == colid
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_val = ones(Float64,length(node_Hrows))
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                return length(mat.rowval)
            elseif colid == 0
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                node_val = ones(Float64,length(node_Hrows))
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                return length(mat.rowval)
            else
                return 0
            end
        else #fill in values
            if rowid == colid
                node_Hmap = node.ext[:Hmap]
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                if colid ==  0
                    local_x = x0
                else
                    local_x = x1
                end
                local_unsym_values = Array{Float64}(undef,local_data.local_unsym_hessnnz)
                node_val = ones(Float64,length(node_Hrows))
                #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
                local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
                local_m_eq = length(local_data.eq_idx)
                local_m_ineq = length(local_data.ineq_idx)
                local_lambda = zeros(Float64, local_data.local_m)
                for i in 1:local_m_eq
                    local_lambda[local_data.eq_idx[i]] = lambda[i]
                end
                for i in 1:local_m_ineq
                    local_lambda[local_data.ineq_idx[i]] = lambda[i+local_m_eq]
                end

                #eval_hesslag(local_data.d, local_unsym_values, local_x, obj_factor*local_scl, local_lambda)
                pips_eval_hessian_lagrangian(local_data.d, local_unsym_values, local_x, obj_factor*local_scl, local_lambda)
                local_sym_index=1
                for i in 1:local_data.local_unsym_hessnnz
                    if node_Hmap[i]
                        node_val[local_sym_index] = local_unsym_values[i]
                        local_sym_index +=1
                    end
                end
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                array_copy(mat.rowval,rowidx)
                array_copy(mat.colptr,colptr)
                original_copy(mat.nzval,values)
            elseif colid ==	0
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_val = zeros(Float64,length(node_Hrows))
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                array_copy(mat.rowval,rowidx)
                array_copy(mat.colptr,colptr)
                original_copy(mat.nzval,values)
            else
            end
        end
        return Int32(1)
    end

    if !MPI.Initialized()
        MPI.Init()
    end
    #println("Loaded Functions")
    comm = MPI.COMM_WORLD
    if(MPI.Comm_rank(comm) == 0)
        t1 = time()
    end
    println(comm)


    #Create FakeModel (The PIPS interface model) and pass all the functions it requires
    model = FakeModel(:Min,0, scen,str_init_x0, str_prob_info, str_eval_f, str_eval_g, str_eval_grad_f, str_eval_jac_g, str_eval_h,str_write_solution)
    #println("Created PIPS Model")
    prob = createProblemStruct(comm, model, true)

    println("Created PIPS-NLP Problem Struct")
    ret = solveProblemStruct(prob)
    println("Solved PIPS-NLP Problem Struct")
    root = 0
    r = MPI.Comm_rank(comm)


    if MPI.Comm_rank(comm) == 0
        println("Timing Results:")
        println("init_x0  ",prob.t_jl_init_x0)
        println("str_init_x0  ", prob.t_jl_str_prob_info)
        println("eval_f  ", prob.t_jl_eval_f)
        println("eval_g0  ",prob.t_jl_eval_g)
        println("eval_grad_f  ", prob.t_jl_eval_grad_f)
        println("eval_jac  ", prob.t_jl_eval_jac_g)
        println("str_eval_jac  ", prob.t_jl_str_eval_jac_g)
        println("eval_h  ",  prob.t_jl_eval_h)
        println("str_eval_h ",  prob.t_jl_str_eval_h)
        println("eval_write_solution  ",  prob.t_jl_write_solution)
        println("PIPS-NLP time:   ",  time() - t1, " (s)")
    end

    for (idx,node) in enumerate(modelList)  #set solution values for each model
        local_data = getData(node)
        if idx != 1
            coreid = zeros(Int, 1)
            sc = MPI.Reduce(local_data.coreid, MPI.SUM, root, comm)
            if r == root
                coreid[1] = sc
            end
            MPI.Bcast!(coreid, length(coreid), root, comm)
            n = zeros(Int, 1)
            n[1] = local_data.n
            MPI.Bcast!(n,      length(n),      coreid[1], comm)
            if r != coreid[1]
                local_data.n = n[1]
                local_data.x_sol = zeros(Float64,local_data.n)
            end
            MPI.Bcast!(local_data.x_sol, local_data.n, coreid[1], comm)

            node.ext[:colVal] = local_data.x_sol
            #node.colVal = local_data.x_sol
        else
            node.ext[:colVal] = local_data.x_sol
            #node.colVal = local_data.x_sol
        end
    end
    status = :Unknown
    if ret == 0
        status = :Optimal
    elseif ret == 1
        status = :Not_Finished
    elseif	ret == 2
        status = :Maximum_Iterations_Exceeded
    elseif	ret == 3
        status = :Infeasible_Problem_Detected
    elseif ret == 4
        status = :Restoration_needed
    else
        status = :UnknownStatus
    end

    return status
end  #end pips_nlp_solve

end #end module
