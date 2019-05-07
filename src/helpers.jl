#############################################
# Helpers
#############################################
function exchange(a,b)
	 temp = a
         a=b
         b=temp
	 return (a,b)
end

function sparseKeepZero(I::AbstractVector{Ti},
    J::AbstractVector{Ti},
    V::AbstractVector{Tv},
    nrow::Integer, ncol::Integer) where {Tv,Ti<:Integer}
    N = length(I)
    if N != length(J) || N != length(V)
        throw(ArgumentError("triplet I,J,V vectors must be the same length"))
    end
    if N == 0
        return spzeros(eltype(V), Ti, nrow, ncol)
    end

    # Work array
    Wj = Array{Ti}(undef,max(nrow,ncol)+1)
    # Allocate sparse matrix data structure
    # Count entries in each row
    Rnz = zeros(Ti, nrow+1)
    Rnz[1] = 1
    nz = 0
    for k=1:N
        iind = I[k]
        iind > 0 || throw(ArgumentError("all I index values must be > 0"))
        iind <= nrow || throw(ArgumentError("all I index values must be ≤ the number of rows"))
        Rnz[iind+1] += 1
        nz += 1
    end
    Rp = cumsum(Rnz)
    Ri = Array{Ti}(undef,nz)
    Rx = Array{Tv}(undef,nz)

    # Construct row form
    # place triplet (i,j,x) in column i of R
    # Use work array for temporary row pointers
    @simd for i=1:nrow; @inbounds Wj[i] = Rp[i]; end
    @inbounds for k=1:N
        iind = I[k]
        jind = J[k]
        jind > 0 || throw(ArgumentError("all J index values must be > 0"))
        jind <= ncol || throw(ArgumentError("all J index values must be ≤ the number of columns"))
        p = Wj[iind]
        Vk = V[k]
        Wj[iind] += 1
        Rx[p] = Vk
        Ri[p] = jind
    end

    # Reset work array for use in counting duplicates
    @simd for j=1:ncol; @inbounds Wj[j] = 0; end

    # Sum up duplicates and squeeze
    anz = 0
    @inbounds for i=1:nrow
        p1 = Rp[i]
        p2 = Rp[i+1] - 1
        pdest = p1
        for p = p1:p2
            j = Ri[p]
            pj = Wj[j]
            if pj >= p1
                Rx[pj] = Rx[pj] + Rx[p]
            else
                Wj[j] = pdest
                if pdest != p
                    Ri[pdest] = j
                    Rx[pdest] = Rx[p]
                end
                pdest += one(Ti)
            end
        end
        Rnz[i] = pdest - p1
        anz += (pdest - p1)
    end

    # Transpose from row format to get the CSC format
    RiT = Array{Ti}(undef,anz)
    RxT = Array{Tv}(undef,anz)

    # Reset work array to build the final colptr
    Wj[1] = 1
    @simd for i=2:(ncol+1); @inbounds Wj[i] = 0; end
    @inbounds for j = 1:nrow
        p1 = Rp[j]
        p2 = p1 + Rnz[j] - 1
        for p = p1:p2
            Wj[Ri[p]+1] += 1
        end
    end
    RpT = cumsum(Wj[1:(ncol+1)])

    # Transpose
    @simd for i=1:length(RpT); @inbounds Wj[i] = RpT[i]; end
    @inbounds for j = 1:nrow
        p1 = Rp[j]
        p2 = p1 + Rnz[j] - 1
        for p = p1:p2
            ind = Ri[p]
            q = Wj[ind]
            Wj[ind] += 1
            RiT[q] = j
            RxT[q] = Rx[p]
        end
    end

    return SparseMatrixCSC(nrow, ncol, RpT, RiT, RxT)
end

#Convert Julia indices to C indices
function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end

#NOTE: These are helper functions to get data from JuMP models into the PIPS-NLP Interface.
#Some functions taken from Ipopt.jl
function numconstraints(m::JuMP.Model)
    num_cons = 0
    constraint_types = JuMP.list_of_constraint_types(m)
    for (func,set) in constraint_types
        if func != JuMP.VariableRef #This is a variable bound, not a PIPS-NLP constraint
            num_cons += JuMP.num_constraints(m,func,set)
        end
    end
    num_cons += JuMP.num_nl_constraints(m)
    return num_cons
end

function variableupperbounds(m::JuMP.Model)
    #Get upper bound variable constraints
    upper_bounds = ones(JuMP.num_variables(m))*Inf #Assume no upper bound by default

    var_bound_constraints = JuMP.all_constraints(m,JuMP.VariableRef,MOI.LessThan{Float64})
    for var_bound_ref in var_bound_constraints
        var_constraint = JuMP.constraint_object(var_bound_ref)
        var = var_constraint.func
        index = var.index
        upper_bound = var_constraint.set.upper
        upper_bounds[index.value] = upper_bound
    end
    #Get fixed variable constraints
    var_equal_constraints = JuMP.all_constraints(m,JuMP.VariableRef,MOI.EqualTo{Float64})
    for var_equal_ref in var_equal_constraints
        var_constraint = JuMP.constraint_object(var_equal_ref)
        var = var_constraint.func
        index = var.index
        upper_bound = var_constraint.set.value
        upper_bounds[index.value] = upper_bound
    end

    #TODO sort?
    return upper_bounds
end

function variablelowerbounds(m::JuMP.Model)
    lower_bounds = ones(JuMP.num_variables(m))*-Inf #Assume no upper bound by default
    var_bound_constraints = JuMP.all_constraints(m,JuMP.VariableRef,MOI.GreaterThan{Float64})
    for var_bound_ref in var_bound_constraints
        var_constraint = JuMP.constraint_object(var_bound_ref)
        var = var_constraint.func
        index = var.index
        lower_bound = var_constraint.set.lower
        lower_bounds[index.value] = lower_bound
    end

    var_equal_constraints = JuMP.all_constraints(m,JuMP.VariableRef,MOI.EqualTo{Float64})
    for var_equal_ref in var_equal_constraints
        var_constraint = JuMP.constraint_object(var_equal_ref)
        var = var_constraint.func
        index = var.index
        lower_bound = var_constraint.set.value
        lower_bounds[index.value] = lower_bound
    end

    #TODO sort?
    return lower_bounds
end

function pips_jacobian_structure(m::JuMP.Model)
end

mutable struct ConstraintData
    linear_le_constraints::Vector{JuMP.ScalarConstraint{GenericAffExpr{Float64,VariableRef},MathOptInterface.LessThan{Float64}}}
    linear_ge_constraints::Vector{JuMP.ScalarConstraint{GenericAffExpr{Float64,VariableRef},MathOptInterface.GreaterThan{Float64}}}
	linear_interval_constraints::Vector{JuMP.ScalarConstraint{GenericAffExpr{Float64,VariableRef},MathOptInterface.Interval{Float64}}}
    linear_eq_constraints::Vector{JuMP.ScalarConstraint{GenericAffExpr{Float64,VariableRef},MathOptInterface.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{JuMP.ScalarConstraint{GenericQuadExpr{Float64,VariableRef},MathOptInterface.LessThan{Float64}}}
    quadratic_ge_constraints::Vector{JuMP.ScalarConstraint{GenericQuadExpr{Float64,VariableRef},MathOptInterface.GreaterThan{Float64}}}
	quadratic_interval_constraints::Vector{JuMP.ScalarConstraint{GenericQuadExpr{Float64,VariableRef},MathOptInterface.Interval{Float64}}}
    quadratic_eq_constraints::Vector{JuMP.ScalarConstraint{GenericQuadExpr{Float64,VariableRef},MathOptInterface.EqualTo{Float64}}}
	nonlinear_constraints::Vector{JuMP._NonlinearConstraint}
end

ConstraintData() = ConstraintData([], [], [], [], [], [],[],[],[])

function get_constraint_data(m::JuMP.Model)
	con_data = ConstraintData()

	constraint_types = JuMP.list_of_constraint_types(m)

    for (func,set) in constraint_types
        if func == JuMP.VariableRef 	#This is a variable bound, not a PIPS-NLP constraint
			continue
		else
	        constraint_refs = JuMP.all_constraints(m, func, set)
	        for constraint_ref in constraint_refs
	            constraint = JuMP.constraint_object(constraint_ref)

				func_type = typeof(constraint.func)
				con_type = typeof(constraint.set)

				if func_type == JuMP.GenericAffExpr{Float64,JuMP.VariableRef}
					if con_type == MOI.LessThan{Float64}
						push!(con_data.linear_le_constraints,constraint)
					elseif con_type == MOI.GreaterThan{Float64}
						push!(con_data.linear_ge_constraints,constraint)
					elseif con_type == MOI.Interval{Float64}
						push!(con_data.linear_interval_constraints,constraint)
					elseif con_type == MOI.EqualTo{Float64}
						push!(con_data.linear_eq_constraints,constraint)
					end
				elseif func_type == JuMP.GenericQuadExpr{Float64,JuMP.VariableRef}
					if con_type == MOI.LessThan{Float64}
						push!(con_data.quadratic_le_constraints,constraint)
					elseif con_type == MOI.GreaterThan{Float64}
						push!(con_data.quadratic_ge_constraints,constraint)
					elseif con_type == MOI.Interval{Float64}
						push!(con_data.quadratic_interval_constraints,constraint)
					elseif con_type == MOI.EqualTo{Float64}
						push!(con_data.quadtratic_eq_constraints,constraint)
					end
	            else
	                error("Could not figure out constraint type for $(constraint_ref)")
	            end
	        end
        end
    end
	if m.nlp_data != nothing
		con_data.nonlinear_constraints = m.nlp_data.nlconstr
	end
	return con_data
end

linear_le_offset(constraints::ConstraintData) = 0
linear_ge_offset(constraints::ConstraintData) = length(constraints.linear_le_constraints)
linear_interval_offset(constraints::ConstraintData) = linear_ge_offset(constraints) + length(constraints.linear_ge_constraints)
linear_eq_offset(constraints::ConstraintData) = linear_interval_offset(constraints) + length(constraints.linear_interval_constraints)
quadratic_le_offset(constraints::ConstraintData) = linear_eq_offset(constraints) + length(constraints.linear_eq_constraints)
quadratic_ge_offset(constraints::ConstraintData) = quadratic_le_offset(constraints) + length(constraints.quadratic_le_constraints)
quadratic_interval_offset(constraints::ConstraintData) =  quadratic_ge_offset(constraints) + length(constraints.quadratic_ge_constraints)
quadratic_eq_offset(constraints::ConstraintData) = quadratic_interval_offset(constraints) + length(constraints.quadratic_interval_constraints)
nlp_constraint_offset(constraints::ConstraintData) = quadratic_eq_offset(constraints) + length(constraints.quadratic_eq_constraints)

function numconstraints(con_data::ConstraintData)
	fields = fieldnames(ConstraintData)
	return sum([length(getfield(con_data,field)) for field in fields])
end

function constraintbounds(con_data::ConstraintData)
	num_cons = numconstraints(con_data)
	# Setup indices.  Need to get number of constraints
	constraint_lower = ones(num_cons)*-Inf
	constraint_upper = ones(num_cons)*Inf

	#linear_le_constraints
	constraints = con_data.linear_le_constraints
	offset = linear_le_offset(con_data)
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.upper for i in 1:length(constraints)]

	#linear_ge_constraints
	constraints = con_data.linear_ge_constraints
	offset = linear_ge_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.lower for i in 1:length(constraints)]

	#linear_interval_constraints
	constraints = con_data.linear_interval_constraints
	offset = linear_interval_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.lower for i in 1:length(constraints)]
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.upper for i in 1:length(constraints)]

	#linear_eq_constraints
	constraints = con_data.linear_eq_constraints
	offset = linear_eq_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.value for i in 1:length(constraints)]
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.value for i in 1:length(constraints)]

	#quadratic_le_constraints
	constraints = con_data.quadratic_le_constraints
	offset = quadratic_le_offset(con_data)
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.upper for i in 1:length(constraints)]

	#quadratic_ge_constraints
	constraints = con_data.quadratic_ge_constraints
	offset = quadratic_ge_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.lower for i in 1:length(constraints)]

	#quadratic_interval_constraints
	constraints = con_data.quadratic_interval_constraints
	offset = quadratic_interval_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.lower for i in 1:length(constraints)]
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.upper for i in 1:length(constraints)]

	#quadratic_eq_constraints
	constraints = con_data.quadratic_eq_constraints
	offset = quadratic_eq_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].set.value for i in 1:length(constraints)]
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].set.value for i in 1:length(constraints)]

	#nonlinear constraints
	constraints = con_data.nonlinear_constraints
	offset = nlp_constraint_offset(con_data)
	constraint_lower[offset + 1:offset + length(constraints)] .= [constraints[i].lb for i in 1:length(constraints)]
	constraint_upper[offset + 1:offset + length(constraints)] .= [constraints[i].ub for i in 1:length(constraints)]

	return constraint_lower,constraint_upper
end

#####################
# JACOBIAN
#####################
function append_to_jacobian_sparsity!(jacobian_sparsity, constraint::JuMP.ScalarConstraint{GenericAffExpr{Float64,VariableRef}}, row)
	aff = constraint.func
    for term in keys(aff.terms)
        push!(jacobian_sparsity, (row, term.index.value))
    end
end

function append_to_jacobian_sparsity!(jacobian_sparsity, constraint::JuMP.ScalarConstraint{GenericQuadExpr{Float64,VariableRef}}, row)
	quad = constraint.func
    for term in keys(quad.aff.terms)
        push!(jacobian_sparsity, (row, term.index.value))
    end
    for term in keys(quad.terms)
        row_idx = term.a.index
        col_idx = term.b.index
        if row_idx == col_idx
            push!(jacobian_sparsity, (row, row_idx.value))
        else
            push!(jacobian_sparsity, (row, row_idx.value))
            push!(jacobian_sparsity, (row, col_idx.value))
        end
    end
end

macro append_to_jacobian_sparsity(array_name)
    escrow = esc(:row)
    quote
        for constraint in $(esc(array_name))
            append_to_jacobian_sparsity!($(esc(:jacobian_sparsity)), constraint, $escrow)
            $escrow += 1
        end
    end
end

function pips_jacobian_structure(d::JuMP.NLPEvaluator)
    num_nlp_constraints = length(d.m.nlp_data.nlconstr)
    if num_nlp_constraints > 0
		MOI.initialize(d,[:Grad,:Jac,:Hess])
        nlp_jacobian_sparsity = MOI.jacobian_structure(d)
    else
        nlp_jacobian_sparsity = []
    end

    jacobian_sparsity = Tuple{Int64,Int64}[]
    row = 1
	con_data = get_constraint_data(d.m)

    @append_to_jacobian_sparsity con_data.linear_le_constraints
    @append_to_jacobian_sparsity con_data.linear_ge_constraints
	@append_to_jacobian_sparsity con_data.linear_interval_constraints
    @append_to_jacobian_sparsity con_data.linear_eq_constraints
    @append_to_jacobian_sparsity con_data.quadratic_le_constraints
    @append_to_jacobian_sparsity con_data.quadratic_ge_constraints
	@append_to_jacobian_sparsity con_data.quadratic_interval_constraints
    @append_to_jacobian_sparsity con_data.quadratic_eq_constraints
    for (nlp_row, column) in nlp_jacobian_sparsity
        push!(jacobian_sparsity, (nlp_row + row - 1, column))
    end
    return jacobian_sparsity
end

#####################
# HESSIAN
#####################
append_to_hessian_sparsity!(hessian_sparsity, ::Union{JuMP.VariableRef,JuMP.GenericAffExpr}) = nothing

function append_to_hessian_sparsity!(hessian_sparsity, quad::JuMP.GenericQuadExpr{Float64,VariableRef})
    for term in keys(quad.terms)
        push!(hessian_sparsity, (term.a.index.value,term.b.index.value))
    end
end

#NOTE: there can be duplicate entries
function pips_hessian_lagrangian_structure(d::JuMP.NLPEvaluator)
    hessian_sparsity = Tuple{Int64,Int64}[]
	m = d.m
	con_data = get_constraint_data(m)
	if m.nlp_data == nothing
	    append_to_hessian_sparsity!(hessian_sparsity, JuMP.objective_function(m))
	end
    for constraint in con_data.quadratic_le_constraints
		quad = constraint.func
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    for constraint in con_data.quadratic_ge_constraints
		quad = constraint.func
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
	for constraint in con_data.quadratic_interval_constraints
		quad = constraint.func
		append_to_hessian_sparsity!(hessian_sparsity, quad)
	end
    for constraint in con_data.quadratic_eq_constraints
		quad = constraint.func
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    nlp_hessian_sparsity = MOI.hessian_lagrangian_structure(d)
    append!(hessian_sparsity, nlp_hessian_sparsity)
    return hessian_sparsity
end

#####################################
# OBJECTIVE FUNCTION EVALUATION
#####################################

function eval_function(var::JuMP.VariableRef, x)
    return x[var.index.value]
end

function eval_function(aff::JuMP.GenericAffExpr{Float64,VariableRef}, x)
    function_value = aff.constant
    for (var,coeff) in aff.terms
        # Note the implicit assumtion that VariableIndex values match up with
        # x indices. This is valid because in this wrapper ListOfVariableIndices
        # is always [1, ..., NumberOfVariables].
        function_value += coeff*x[var.index.value]
    end
    return function_value
end

function eval_function(quad::JuMP.GenericQuadExpr{Float64,VariableRef}, x)
    function_value = quad.aff.constant
    for term in quad.affine_terms
        function_value += term.coefficient*x[term.variable_index.value]
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            function_value += 0.5*coefficient*x[row_idx.value]*x[col_idx.value]
        else
            function_value += coefficient*x[row_idx.value]*x[col_idx.value]
        end
    end
    return function_value
end

function eval_objective(model::Optimizer, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return eval_function(model.objective, x)
    else
        # No objective function set. This could happen with FEASIBILITY_SENSE.
        return 0.0
    end
end


# function constraintbounds(m::JuMP.Model)
#     num_cons = numconstraints(m)
#     # Setup indices.  Need to get number of constraints
#     # constraint_lower = ones(num_cons)*-Inf
#     # constraint_upper = ones(num_cons)*Inf
#     constraint_indices = Int64[]
#     constraint_lower = Float64[]
#     constraint_upper = Float64[]
#
#     constraint_types = JuMP.list_of_constraint_types(m)
#
#     #Figure out constraint order
#     for (func,set) in constraint_types
#         if func != JuMP.VariableRef #This is a variable bound, not a PIPS-NLP constraint
#             constraint_refs = JuMP.all_constraints(m, func, set)
#             for constraint_ref in constraint_refs
#                 constraint_index = constraint_ref.index  #moi index
#                 push!(constraint_indices,constraint_index.value)
#                 constraint = JuMP.constraint_object(constraint_ref)
#                 con_type = typeof(constraint.set)
#                 if con_type == MOI.LessThan{Float64}
#                     push!(constraint_upper,constraint.set.upper)
#                     push!(constraint_lower,-Inf)
#                     #constraint_upper[constraint_index.value] = constraint.set.upper
#                 elseif con_type == MOI.GreaterThan{Float64}
#                     push!(constraint_upper,Inf)
#                     push!(constraint_lower,constraint.set.lower)
#                     #constraint_lower[constraint_index.value] = constraint.set.lower
#                 elseif con_type == MOI.Interval{Float64}
#                     push!(constraint_upper,constraint.set.upper)
#                     push!(constraint_lower,constraint.set.lower)
#                     # constraint_upper[constraint_index.value] = constraint.set.upper
#                     # constraint_lower[constraint_index.value] = constraint.set.lower
#                 elseif con_type == MOI.EqualTo{Float64}
#                     push!(constraint_upper,constraint.set.value)
#                     push!(constraint_lower,constraint.set.value)
#                     # constraint_upper[constraint_index.value] = constraint.set.value
#                     # constraint_lower[constraint_index.value] = constraint.set.value
#                 else
#                     error("Could not figure out constraint type for $(constraint_ref) to get bounds")
#                 end
#             end
#         end
#     end
#     #sort constraint bounds by smallest to largest index
#     p = sortperm(constraint_indices)
#     constraint_lower = constraint_lower[p]
#     constraint_upper = constraint_upper[p]
#
#     #Now add nonlinear constraint bounds
#     if m.nlp_data != nothing
#         for nl_constr in m.nlp_data.nlconstr
#             push!(constraint_lower,nl_constr.lb)
#             push!(constraint_upper,nl_constr.ub)
#         end
#     end
#     @assert length(constraint_lower) == length(constraint_upper) == num_cons
#     return constraint_lower,constraint_upper
# end
