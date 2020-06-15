# time varying operators


####################################################
#
# OpPathScalar is used to hold (t, p_t⋅∇)
# mOpᵀPathScalar also holds (t, p_t⋅∇) but when applied operates with -(p_t⋅∇)ᵀ
#
######################################################

# Models time sequence of sparse operators op_t for t in tspan01
# (t, op_t) ≡ (tspan01[i], op[i]) for time indices i ∈ 1:length(tspan01)
struct OpPathScalar{T<:Real,In<:Integer}
	tspan01::Vector{T}
	op::Vector{SparseMatrixCSC{T,In}}
end

struct mOpᵀPathScalar{T<:Real,In<:Integer} 
	tspan01::Vector{T}
	op::Vector{SparseMatrixCSC{T,In}}
end

# conversion between the two types ... used in adjoint method
mOpᵀPathScalar{T,In}(opp::OpPathScalar{T,In}) where {T<:Real,In<:Integer} =  mOpᵀPathScalar{T,In}(opp.tspan01, opp.op)
OpPathScalar{T,In}(opp::mOpᵀPathScalar{T,In}) where {T<:Real,In<:Integer} =  OpPathScalar{T,In}(opp.tspan01, opp.op)


# fills a preallocated velocity 𝒱_fx with opp_t * fx ... using in flow
@inbounds function (opp::OpPathScalar{T,In})(𝒱_fx::A, t::T, fx::A) where {T<:Real,In<:Integer,A<:Vector{Vector{T}}}
    t_indx = find_t_in_tspan(t, opp.tspan01)
    Base.Threads.@threads for f ∈ 1:length(fx)
        mul!(𝒱_fx[f], opp.op[t_indx], fx[f]) 
    end
    return nothing
end

# fills a preallocated velocity 𝒱_fx with -(opp_t)ᵀ * fx ... used in transpose flow
@inbounds function (opp::mOpᵀPathScalar{T,In})(𝒱_fx::A, t::T, fx::A) where {T<:Real,In<:Integer,A<:Vector{Vector{T}}}
    t_indx = find_t_in_tspan(t, opp.tspan01)
    Base.Threads.@threads for f ∈ 1:length(fx)
        mul!(𝒱_fx[f], transpose(opp.op[t_indx]), fx[f]) 
        𝒱_fx[f] .*= -1
    end
    return nothing
end



#= ###################################################


OpPathScalar is used to hold (t, X_t) and (t, Y_t) where
      X_t[q] = sum( M_t[q,i] * ∇[i] for i=1:2 )
      Y_t[q] = sum( - t * vx[p] * Mtx[p,i] * Mtx[j,q] * ∇∇[i,j] for i,j,p=1:2 )
whos action is given by 
      sum( (∇[q]*f) * (X_t[q]) for q=1:2 ) * δϕ
      sum( (∇[q]*f) * (Y_t[q]) for q=1:2 ) * δϕ


◮ Note: the transpose delta flow is given by 
      sum{ sum( transpose(-X_t[q]) * (∇[q]*f) for q=1:2) * δf for f ∈ (t,q,u) }  
      sum{ sum( transpose(-Y_t[q]) * (∇[q]*f) for q=1:2) * δf for f ∈ (t,q,u) }  

◮ Note: ∇[i] in X_t should be the covariant ∇^i derivative used for ϕ.

◮ Note: ∇∇[i,j] in Y_t should be ∇_i∇^j.

=# #####################################################


# Models time sequence of sparse operators X_t^q, Y_t^q for t in tspan01
# (t, X_t^q) ≡ (tspan01[i], op[i][q]) for time indices i ∈ 1:length(tspan01)
struct OpPathVector{T<:Real,In<:Integer}
	tspan01::Vector{T}
	op::Vector{Vector{SparseMatrixCSC{T,In}}}
end


####################################################

# used to construct operator path for delta transpose flow

####################################################


@inbounds function get_Xhalf_Yhalf_p∇full(ϕx::Vector{T}, ∂i::∂{T,In}, ϕ∂i::∂{T,In}, ∂i∂j::∂∂{T,In}, ode_steps::Int = default_ode_steps) where {T<:Real, In<:Integer}
    n_pix = length(ϕx)
    vx    = Vector{Vector{T}}(undef, 2)
    vx[1], vx[2] = ϕ∂i(ϕx)
    tspan01       = T.(runge_kutta_times(ode_steps))
    tspan01_half  = tspan01[1:2:end]
    Mx     = get_Mx(vx, ∂i, tspan01)
    px     = get_px(vx, Mx, tspan01)
    px∇    = get_px∇(px, ∂i)
    Xx, Yx = get_Xx_Yx(vx, Mx[1:2:end], ϕ∂i, ∂i∂j, tspan01_half) # note: ϕ∂i here instead of ∂i
    Xhalf  = OpPathVector{T,In}(tspan01_half, Xx)  
    Yhalf  = OpPathVector{T,In}(tspan01_half, Yx) 
    p∇     = OpPathScalar{T,In}(tspan01, px∇)
    return Xhalf, Yhalf, p∇
end


######  helpers #########

@inbounds function get_Xx_Yx(vx, Mx_half, ϕ∂i::∂{T,In}, ∂iϕ∂j::∂∂{T,In}, tspan01_half) where {T<:Real, In<:Integer}
	px′_half     = get_px′(vx, Mx_half, tspan01_half)
    Xx           = Vector{Vector{SparseMatrixCSC{T,In}}}(undef, length(tspan01_half)) 
    Yx           = Vector{Vector{SparseMatrixCSC{T,In}}}(undef, length(tspan01_half)) 
	# for i in 1:length(tspan01_half)
    Base.Threads.@threads for i in 1:length(tspan01_half)
        Xx[i], Yx[i] = get_Xx_Yx_at_t_helper(Mx_half[i], px′_half[i], ϕ∂i, ∂iϕ∂j, tspan01_half[i])
    end
    return Xx, Yx
end
@inbounds function get_Xx_Yx_at_t_helper(Mtx::Matrix{Vector{T}}, ptx′::Vector{Vector{T}}, ϕ∂i::∂{T,In}, ∂iϕ∂j::∂∂{T,In}, t::T) where {T<:Real, In<:Integer}
    Xtx = SparseMatrixCSC{T,In}[zero(ϕ∂i.∇[1]), zero(ϕ∂i.∇[1])]
    Ytx = SparseMatrixCSC{T,In}[zero(∂iϕ∂j.∇∇[1,1]), zero(∂iϕ∂j.∇∇[1,1])]
    r∇  = rowvals(ϕ∂i.∇[1])      # NOTE: assuming rowvals(∇1) == rowvals(∇2) by construction of  ∇1, ∇2
    r∇∇ = rowvals(∂iϕ∂j.∇∇[1,1]) # NOTE: assuming rowvals(∇1) == rowvals(∇2) by construction of  ∇1, ∇2
    for ir in 1:length(r∇)
        r∇ir = r∇[ir]
        Xtx[1].nzval[ir] += Mtx[1,1][r∇ir] * ϕ∂i.∇[1].nzval[ir] + Mtx[1,2][r∇ir] * ϕ∂i.∇[2].nzval[ir]
        Xtx[2].nzval[ir] += Mtx[2,1][r∇ir] * ϕ∂i.∇[1].nzval[ir] + Mtx[2,2][r∇ir] * ϕ∂i.∇[2].nzval[ir]
    end
    for ir in 1:length(r∇∇)
        r∇∇ir = r∇∇[ir]
        for i=1:2, j=1:2
        	Ytx[1].nzval[ir] -= t * ptx′[i][r∇∇ir] * Mtx[j,1][r∇∇ir] * ∂iϕ∂j.∇∇[i,j].nzval[ir]
        	Ytx[2].nzval[ir] -= t * ptx′[i][r∇∇ir] * Mtx[j,2][r∇∇ir] * ∂iϕ∂j.∇∇[i,j].nzval[ir]
    	end
    end
    return Xtx, Ytx
end

@inbounds function get_px′(vx::Vector{Vector{T}}, Mx::Vector{Matrix{Vector{T}}}, tspan01_half::Vector{T}) where {T<:Real, In<:Integer}
    n_pix = length(vx[1])
    n_rk  = length(tspan01_half)
    px′    = Vector{Vector{Vector{T}}}(undef, n_rk)   # px[indx_tspan01_half][2] == (px_t)_{2}
    #for i in 1:n_rk
    Base.Threads.@threads for i in 1:n_rk
        px′[i]   = Vector{T}[fill(T(0), n_pix) for ii ∈ 1:2]
        for cm = 1:n_pix
            px′[i][1][cm] = vx[1][cm] * Mx[i][1,1][cm] + vx[2][cm] * Mx[i][2,1][cm]
            px′[i][2][cm] = vx[1][cm] * Mx[i][1,2][cm] + vx[2][cm] * Mx[i][2,2][cm]
        end
    end
    return px′
end
