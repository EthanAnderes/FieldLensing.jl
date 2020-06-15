
#################################################################
#
# Field requirements:  <: AbstractMapFlow requires
#   tstart::Int
#   tstop::Int
#   opp::SparseOpPathScalar{T,In} 
#
# Method requirements: must define inv and adjoint
#
#######################################################

abstract type AbstractMapFlow{T<:Real,In<:Integer} end

*(L::AbstractMapFlow{T,In}, f) where {T<:Real,In<:Integer} = flow(L, f)
\(L::AbstractMapFlow{T,In}, f) where {T<:Real,In<:Integer} = flow(inv(L),  f)


#################################

# types MapFlow and MapFlowᵀ

################################

struct MapFlow{T<:Real, In<:Integer} <: AbstractMapFlow{T,In}
    tstart::Int
    tstop::Int
    opp::OpPathScalar{T,In} 
end
# note: opp.tspan01 is twice the resolution of the lensing path. 
# note: opp holds p∇

struct MapFlowᵀ{T<:Real, In<:Integer} <: AbstractMapFlow{T,In}
    tstart::Int
    tstop::Int
    opp::mOpᵀPathScalar{T,In}
end
# note: opp still holds p∇ but the mOpᵀPathScalar wrapper means that it has a modified action



#################################

# constructors (note: the typical constructor of MapFlowᵀ is with adjoint(MapFlow)

################################


function MapFlow{T,In}(ϕ::Field{P,T,S0}, ∂i::∂{T,In}, ϕ∂i::∂{T,In}, tstart::Int, tstop::Int, ode_steps::Int = default_ode_steps) where {T<:Real, In<:Integer, P<:Pix}
    ϕx, = ϕ |> LenMapBasis |> data |> deepcopy
    return MapFlow{T,In}(vec(ϕx), ∂i, ϕ∂i, tstart, tstop, ode_steps)
end

function MapFlow{T,In}(ϕx::Vector{T}, ∂i::∂{T,In}, ϕ∂i::∂{T,In}, tstart::Int, tstop::Int, ode_steps::Int = default_ode_steps) where {T<:Real, In<:Integer}
    n_pix = length(ϕx)
    vx    = Vector{Vector{T}}(undef, 2)
    vx[1], vx[2] = ϕ∂i(ϕx)
    tspan01   = T.(runge_kutta_times(ode_steps))    
    Mx  = get_Mx(vx, ∂i, tspan01)
    px  = get_px(vx, Mx, tspan01)
    px∇ = get_px∇(px, ∂i)
    p∇  = OpPathScalar{T,In}(tspan01, px∇)
    return MapFlow{T,In}(tstart, tstop, p∇)
end
# Note: ode_steps = length(L.opp.tspan01[1:2:end])-1
 


@inbounds function get_Mx(vx::Vector{Vector{T}}, ∂i::∂{T,In}, tspan01::Vector{T}) where {T<:Real, In<:Integer}
    n_pix  = length(vx[1])
    n_rk   = length(tspan01)
    Mx    = Vector{Matrix{Vector{T}}}(undef, n_rk)   # Mx[indx_tspan01][1,2] == (invMx_t)_{1,2}
    detM  = Vector{Vector{T}}(undef, n_rk)
    ∂1_v1x, ∂2_v1x = ∂i(vx[1])
    ∂1_v2x, ∂2_v2x = ∂i(vx[2])
    Base.Threads.@threads for i in 1:n_rk
        Mx[i]   = Vector{T}[fill(T(0), n_pix) for ii ∈ 1:2, jj ∈ 1:2]
        detM[i] = fill(T(0), n_pix)
        for cm = 1:n_pix
            Mx[i][1,1][cm]  =    1 + tspan01[i] * ∂2_v2x[cm]
            Mx[i][1,2][cm]  =      - tspan01[i] * ∂1_v2x[cm]
            Mx[i][2,1][cm]  =      - tspan01[i] * ∂2_v1x[cm]
            Mx[i][2,2][cm]  =    1 + tspan01[i] * ∂1_v1x[cm]
            detM[i][cm]     =  Mx[i][1,1][cm] * Mx[i][2,2][cm] - Mx[i][1,2][cm]*Mx[i][2,1][cm]
            Mx[i][1,1][cm] = Mx[i][1,1][cm] / detM[i][cm]
            Mx[i][1,2][cm] = Mx[i][1,2][cm] / detM[i][cm]
            Mx[i][2,2][cm] = Mx[i][2,2][cm] / detM[i][cm]
            Mx[i][2,1][cm] = Mx[i][2,1][cm] / detM[i][cm]
        end
    end
    return Mx
end

 
@inbounds function get_px(vx::Vector{Vector{T}}, Mx::Vector{Matrix{Vector{T}}}, tspan01::Vector{T}) where {T<:Real}
    n_pix = length(vx[1])
    n_rk  = length(tspan01)
    px    = Vector{Vector{Vector{T}}}(undef, n_rk)   # px[indx_tspan01][2] == (px_t)_{2}
    # for i in 1:n_rk
    Base.Threads.@threads for i in 1:n_rk
        px[i]   = Vector{T}[fill(T(0), n_pix) for ii ∈ 1:2]
        for cm = 1:n_pix
            px[i][1][cm] = Mx[i][1,1][cm] * vx[1][cm] + Mx[i][1,2][cm] * vx[2][cm]
            px[i][2][cm] = Mx[i][2,1][cm] * vx[1][cm] + Mx[i][2,2][cm] * vx[2][cm]
        end
    end
    return px
end


@inbounds function get_px∇(px::Vector{Vector{Vector{T}}}, ∂i::FieldFlows.∂{T,In}) where {T<:Real, In<:Integer}
    n_rk = length(px)
    px∇   = Vector{SparseMatrixCSC{T,In}}(undef, n_rk) 
    # for i in 1:n_rk
    Base.Threads.@threads for i in 1:n_rk
        px∇[i] = get_px∇_helper(px[i][1], px[i][2], ∂i.∇[1], ∂i.∇[2])
    end
    return px∇
end

@inbounds function get_px∇_helper(p1, p2, ∇1, ∇2)
    p∇ = zero(∇1)
    rowvals∇ = rowvals(∇1) #NOTE: assuming rowvals(∇1) == rowvals(∇2) by construction of  ∇1, ∇2
    # for i in 1:length(rowvals∇)
    Base.Threads.@threads for i in 1:length(rowvals∇)
        p∇.nzval[i] = p1[rowvals∇[i]] * ∇1.nzval[i] + p2[rowvals∇[i]] * ∇2.nzval[i]
    end
    return p∇
end


##########################################

# additional transformation constructors

###############################################


function adjoint(L::MapFlow{T,In}) where {T<:Real,In<:Integer}
    return MapFlowᵀ{T,In}(L.tstop, L.tstart, mOpᵀPathScalar{T,In}(L.opp))
end

function adjoint(L::MapFlowᵀ{T,In}) where {T<:Real,In<:Integer}
    return MapFlow{T,In}(L.tstop, L.tstart, OpPathScalar{T,In}(L.opp))
end



#################################

# impliment `inv` methods

################################

function inv(L::MapFlow{T,In}) where {T<:Real,In<:Integer}
    return MapFlow{T,In}(L.tstop, L.tstart, L.opp)
end

function inv(L::MapFlowᵀ{T,In}) where {T<:Real,In<:Integer}
    return MapFlowᵀ{T,In}(L.tstop, L.tstart, L.opp)
end




#################################################################
#
# flow method using 𝒱op!
#
#######################################################


@inbounds function flow(L::AbstractMapFlow{T,In}, f::F) where {T<:Real, In<:Integer,F<:Vector{Vector{T}}}
    if L.tstart == L.tstop
        return f
    else
        n_pix = length(f[1])
        fx = deepcopy(f)
        n_fields = length(fx)

        𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Vector{T}[fill(T(0), n_pix) for i∈1:n_fields], 5)
                
        ode_steps = length(L.opp.tspan01[1:2:end])-1
        rk_tspan, ϵ = runge_kutta_times(ode_steps, L.tstart, L.tstop, T)
        len_tspan = rk_tspan[1:2:end]
        for i = 1:length(len_tspan)-1
            t = len_tspan[i]

            L.opp(𝒱_fx1, t, fx)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx1[j] .* (ϵ/2)
            end

            L.opp(𝒱_fx2, t + ϵ/2, fx_tmp)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx2[j] .* (ϵ/2)
            end

            L.opp(𝒱_fx3, t + ϵ/2, fx_tmp)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx3[j] .* ϵ
            end

            L.opp(𝒱_fx4, t + ϵ, fx_tmp)
            for j = 1:n_fields
                fx[j] .+= 𝒱_fx1[j].*(ϵ/6) .+ 𝒱_fx2[j].*(ϵ/3) .+ 𝒱_fx3[j].*(ϵ/3) .+ 𝒱_fx4[j].*(ϵ/6)
            end
        end
        return fx
    end
end




@inbounds function flow_path01(L::AbstractMapFlow{T,In}, f0::F) where {T<:Real, In<:Integer,F<:Vector{Vector{T}}}
    # NOTE: f0 must be defined at time 0!
    # NOTE: the returned flow path on L.opp.tspan01[1:2:end]

    tspan01      = L.opp.tspan01
    tspan01_half = L.opp.tspan01[1:2:end]
    ϵ = tspan01_half[2] - tspan01_half[1]

    # constructing fx_path, fx_path[time_index][field][pixel]
    n_fields   = length(f0)
    n_pix      = length(f0[1])
    fx_path    = Vector{Vector{Vector{T}}}(undef, length(tspan01_half)) 
    fx_path[1] = Vector{T}[copy(ff) for ff ∈ f0]
    for i=2:length(tspan01_half)
        fx_path[i] = Vector{T}[Vector{T}(undef, n_pix) for ii ∈ 1:n_fields]
    end

    𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Vector{T}[fill(T(0), n_pix) for i∈1:n_fields], 5)            
    for i = 1:length(tspan01_half)-1
        t = tspan01_half[i]

        L.opp(𝒱_fx1, t, fx_path[i])
        for j = 1:n_fields
            fx_tmp[j] .= fx_path[i][j] .+ 𝒱_fx1[j] .* (ϵ/2)
        end

        L.opp(𝒱_fx2, t + ϵ/2, fx_tmp)
        for j = 1:n_fields
            fx_tmp[j] .= fx_path[i][j] .+ 𝒱_fx2[j] .* (ϵ/2)
        end

        L.opp(𝒱_fx3, t + ϵ/2, fx_tmp)
        for j = 1:n_fields
            fx_tmp[j] .= fx_path[i][j] .+ 𝒱_fx3[j] .* ϵ
        end

        L.opp(𝒱_fx4, t + ϵ, fx_tmp)
        for j = 1:n_fields
            fx_path[i+1][j] .= fx_path[i][j] .+ 𝒱_fx1[j].*(ϵ/6) .+ 𝒱_fx2[j].*(ϵ/3) .+ 𝒱_fx3[j].*(ϵ/3) .+ 𝒱_fx4[j].*(ϵ/6)
        end
    end
    return fx_path, tspan01_half
end

