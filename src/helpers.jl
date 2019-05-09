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
# JACOBIAN STRUCTURE
#####################
function append_to_jacobian_sparsity!(jacobian_sparsity, func::JuMP.GenericAffExpr{Float64,VariableRef}, row)
	aff = func
    for term in keys(aff.terms)
        push!(jacobian_sparsity, (row, term.index.value))
    end
end

function append_to_jacobian_sparsity!(jacobian_sparsity, func::JuMP.GenericQuadExpr{Float64,VariableRef}, row)
	quad = func
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
			func = constraint.func
            append_to_jacobian_sparsity!($(esc(:jacobian_sparsity)), func, $escrow)
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

function pips_jacobian_structure(m::JuMP.Model)
end

#####################
# HESSIAN STRUCTURE
#####################
function has_nl_objective(m::JuMP.Model)
	if m.nlp_data == nothing
		return false
	elseif m.nlp_data.nlobj != nothing
		return true
	else
		return false
	end
end

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
	elseif m.nlp_data.nlobj == nothing
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
#x is an array of variable values returned from the solver
function eval_function(var::JuMP.VariableRef, x)
    return x[var.index.value]
end

function eval_function(aff::JuMP.GenericAffExpr{Float64,VariableRef}, x)
    function_value = aff.constant
    for (var,coeff) in aff.terms
        # NOTE the implicit assumtion that VariableIndex values match up with
        # x indices. This is valid because in this wrapper ListOfVariableIndices
        # is always [1, ..., NumberOfVariables].
        function_value += coeff*x[var.index.value]
    end
    return function_value
end

function eval_function(quad::JuMP.GenericQuadExpr{Float64,VariableRef}, x)
    function_value = quad.aff.constant
    for (var,coeff) in quad.aff.terms
        function_value += coeff*x[var.index.value]
    end
    for (terms,coeff) in quad.terms
        row_idx = terms.a.index
        col_idx = terms.b.index
		function_value += coeff*x[row_idx.value]*x[col_idx.value]
    end
    return function_value
end

function pips_eval_objective(d::JuMP.NLPEvaluator, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
	m = d.m
    if has_nl_objective(m)
        return MOI.eval_objective(d, x)
    elseif JuMP.objective_function(m) !== nothing
        return eval_function(JuMP.objective_function(m), x)
    else
        return 0.0 # No objective function set. This could happen with FEASIBILITY_SENSE.
    end
end

#####################################
# OBJECTIVE GRADIENT EVALUATION
#####################################
function fill_gradient!(grad, x, var::JuMP.VariableRef)
    fill!(grad, 0.0)
    grad[var.index.value] = 1.0
end

function fill_gradient!(grad, x, aff::JuMP.GenericAffExpr{Float64,VariableRef})
    fill!(grad, 0.0)
	for	(var,coeff) in aff.terms
        grad[var.index.value] += coeff
    end
end

function fill_gradient!(grad, x, quad::JuMP.GenericQuadExpr{Float64,VariableRef})
    fill!(grad, 0.0)
    for	(var,coeff) in quad.aff.terms
        grad[var.index.value] += coeff
    end
    #for term in quad.quadratic_terms
	for (terms,coeff) in quad.terms
        row_idx = terms.a.index
        col_idx = terms.b.index
        if row_idx == col_idx
            grad[row_idx.value] += 2*coeff*x[row_idx.value]
        else
            grad[row_idx.value] += coeff*x[col_idx.value]
            grad[col_idx.value] += coeff*x[row_idx.value]
        end
    end
end

function pips_eval_objective_gradient(d::JuMP.NLPEvaluator, grad, x)
	m = d.m
    if has_nl_objective(m)
        MOI.eval_objective_gradient(d, grad, x)
    elseif JuMP.objective_function(m) !== nothing
        fill_gradient!(grad, x, JuMP.objective_function(m))
    else
        fill!(grad, 0.0)
    end
    return
end

#####################################
# CONSTRAINT EVALUATION
#####################################
# Refers to local variables in eval_constraint() below.
macro eval_function(array_name)
    escrow = esc(:row)
    quote
        for constraint in $(esc(array_name))
			func = constraint.func
            $(esc(:g))[$escrow] = eval_function(func, $(esc(:x)))
            $escrow += 1
        end
    end
end

function pips_eval_constraint(d::JuMP.NLPEvaluator, g, x)
    row = 1
    @eval_function con_data.linear_le_constraints
    @eval_function con_data.linear_ge_constraints
	@eval_function con_data.linear_interval_constraints
    @eval_function con_data.linear_eq_constraints
    @eval_function con_data.quadratic_le_constraints
    @eval_function con_data.quadratic_ge_constraints
	@eval_function con_data.quadratic_interval_constraints
    @eval_function con_data.quadratic_eq_constraints
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(d, nlp_g, x)
    return
end

#####################################
# JACOBIAN EVALUATION
#####################################
function fill_constraint_jacobian!(jac_values, start_offset, x, aff::JuMP.GenericAffExpr{Float64,VariableRef})
    num_coefficients = length(aff.terms)
    #for i in 1:num_coefficients
	i = 1
	for coeff in values(aff.terms) #ordered dictionary
        jac_values[start_offset+i] = coeff
		i += 1
    end
    return num_coefficients
end

function fill_constraint_jacobian!(jac_values, start_offset, x, quad::JuMP.GenericQuadExpr{Float64,VariableRef})
	aff = quad.aff
    num_affine_coefficients = length(aff.terms)
	i = 1
	for coeff in values(aff.terms)
        jac_values[start_offset+i] = coeff #quad.affine_terms[i].coefficient
		i += 1
    end
    num_quadratic_coefficients = 0

	for (terms,coeff) in quad.terms
        row_idx = terms.a.index
        col_idx = terms.b.index
        if row_idx == col_idx
            jac_values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = 2*coeff*x[row_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            jac_values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coeff*x[col_idx.value]
            jac_values[start_offset+num_affine_coefficients+num_quadratic_coefficients+2] = coeff*x[row_idx.value]
            num_quadratic_coefficients += 2
        end
    end
    return num_affine_coefficients + num_quadratic_coefficients
end

# Refers to local variables in eval_constraint_jacobian() below.
macro fill_constraint_jacobian(array_name)
    esc_offset = esc(:offset)
    quote
        for constraint in $(esc(array_name))
			func = constraint.func
            $esc_offset += fill_constraint_jacobian!($(esc(:jac_values)),
                                                     $esc_offset, $(esc(:x)),
                                                     func)
        end
    end
end

function pips_eval_constraint_jacobian(d::JuMP.NLPEvaluator, jac_values, x)
    offset = 0
	m = d.m
	con_data = m.ext[:constraint_data]
    @fill_constraint_jacobian con_data.linear_le_constraints
    @fill_constraint_jacobian con_data.linear_ge_constraints
	@fill_constraint_jacobian con_data.linear_interval_constraints
    @fill_constraint_jacobian con_data.linear_eq_constraints
    @fill_constraint_jacobian con_data.quadratic_le_constraints
    @fill_constraint_jacobian con_data.quadratic_ge_constraints
	@fill_constraint_jacobian con_data.quadratic_interval_constraints
    @fill_constraint_jacobian con_data.quadratic_eq_constraints

    nlp_values = view(jac_values, 1+offset:length(jac_values))
    MOI.eval_constraint_jacobian(d, nlp_values, x)
    return
end

##########################################
# HESSIAN OF THE LAGRANGIAN EVALUATION
#########################################
function fill_hessian_lagrangian!(hess_values, start_offset, scale_factor,::Union{JuMP.VariableRef,JuMP.GenericAffExpr{Float64,VariableRef},Nothing})
    return 0
end

function fill_hessian_lagrangian!(hess_values, start_offset, scale_factor, quad::JuMP.GenericQuadExpr{Float64,VariableRef})
	i = 1

	for (terms,coeff) in quad.terms
		row_idx = terms.a.index
		col_idx = terms.b.index
		if row_idx == col_idx
			hess_values[start_offset + i] = 2*scale_factor*coeff
		else
			hess_values[start_offset + i] = scale_factor*coeff
		end
		i += 1
	# for coeff in values(quad.terms)
    #     hess_values[start_offset + i] = scale_factor*coeff
	# 	i += 1
    # end
	end
    return length(quad.terms)
end

function pips_eval_hessian_lagrangian(d::JuMP.NLPEvaluator, hess_values, x, obj_factor, lambda)
    offset = 0
	m = d.m
	con_data = m.ext[:constraint_data]
	if !(has_nl_objective(m))
        offset += fill_hessian_lagrangian!(hess_values, 0, obj_factor, JuMP.objective_function(m))
    end
    for (i, constraint) in enumerate(con_data.quadratic_le_constraints)
		quad = constraint.func
        offset += fill_hessian_lagrangian!(hess_values, offset, lambda[i+quadratic_le_offset(con_data)], quad)
    end
    for (i, constraint) in enumerate(con_data.quadratic_ge_constraints)
		quad = constraint.func
        offset += fill_hessian_lagrangian!(hess_values, offset, lambda[i+quadratic_ge_offset(con_data)], quad)
    end
	for (i, constraint) in enumerate(con_data.quadratic_interval_constraints)
		quad = constraint.func
        offset += fill_hessian_lagrangian!(hess_values, offset, lambda[i+quadratic_interval_offset(con_data)], quad)
    end
    for (i, constraint) in enumerate(con_data.quadratic_eq_constraints)
		quad = constraint.func
        offset += fill_hessian_lagrangian!(hess_values, offset, lambda[i+quadratic_eq_offset(con_data)], quad)
    end
    nlp_values = view(hess_values, 1 + offset : length(hess_values))
    nlp_lambda = view(lambda, 1 + nlp_constraint_offset(con_data) : length(lambda))
    MOI.eval_hessian_lagrangian(d, nlp_values, x, obj_factor, nlp_lambda)
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
