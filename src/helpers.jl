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

function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end



#TODO
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

function constraintbounds(m::JuMP.Model)
    num_cons = numconstraints(m)
    #Setup indices.  Need to get number of constraints
    # constraint_lower = ones(num_cons)*-Inf
    # constraint_upper = ones(num_cons)*Inf
    constraint_indices = Int64[]
    constraint_lower = Float64[]
    constraint_upper = Float64[]

    constraint_types = JuMP.list_of_constraint_types(m)

    #Figure out constraint order
    for (func,set) in constraint_types
        if func != JuMP.VariableRef #This is a variable bound, not a PIPS-NLP constraint
            constraint_refs = JuMP.all_constraints(m, func, set)
            for constraint_ref in constraint_refs
                constraint_index = constraint_ref.index  #moi index
                push!(constraint_indices,constraint_index.value)
                constraint = JuMP.constraint_object(constraint_ref)
                con_type = typeof(constraint.set)
                if con_type == MOI.LessThan{Float64}
                    push!(constraint_upper,constraint.set.upper)
                    push!(constraint_lower,-Inf)
                    #constraint_upper[constraint_index.value] = constraint.set.upper
                elseif con_type == MOI.GreaterThan{Float64}
                    push!(constraint_upper,Inf)
                    push!(constraint_lower,constraint.set.lower)
                    #constraint_lower[constraint_index.value] = constraint.set.lower
                elseif con_type == MOI.Interval{Float64}
                    push!(constraint_upper,constraint.set.upper)
                    push!(constraint_lower,constraint.set.lower)
                    # constraint_upper[constraint_index.value] = constraint.set.upper
                    # constraint_lower[constraint_index.value] = constraint.set.lower
                elseif con_type == MOI.EqualTo{Float64}
                    push!(constraint_upper,constraint.set.value)
                    push!(constraint_lower,constraint.set.value)
                    # constraint_upper[constraint_index.value] = constraint.set.value
                    # constraint_lower[constraint_index.value] = constraint.set.value
                else
                    error("Could not figure out constraint type for $(constraint_ref) to get bounds")
                end
            end
        end
    end
    #sort constraint bounds by smallest to largest index
    p = sortperm(constraint_indices)
    constraint_lower = constraint_lower[p]
    constraint_upper = constraint_upper[p]

    #Now add nonlinear constraint bounds
    if m.nlp_data != nothing
        for nl_constr in m.nlp_data.nlconstr
            push!(constraint_lower,nl_constr.lb)
            push!(constraint_upper,nl_constr.ub)
        end
    end
    @assert length(constraint_lower) == length(constraint_upper) == num_cons
    return constraint_lower,constraint_upper
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
