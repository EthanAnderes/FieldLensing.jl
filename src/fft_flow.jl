# TODO: the catched time path and corresponding RK solver time path is cluncky. 
# TODO: merge all the RK4 code to reduce code duplication. 

#################################

# AbstractFlow

################################

abstract type AbstractFlow{P<:Pix,T<:Real} end

flow(L::AbstractFlow{P,T}, f::Field{P,T}) where {P<:Pix,T<:Real} = error("no method flow(<:AbstractFlow{P,T}, <:Field{P,T}) found")

*(L::AbstractFlow{P,T}, f::Field{P,T}) where {P<:Pix,T<:Real} = flow(L, f)

\(L::AbstractFlow{P,T}, f::Field{P,T}) where {P<:Pix,T<:Real} = flow(inv(L), f)


#################################

# struct definitions Flowϕ, Flowϕᵀ and FlowPathContainer

################################


struct FlowPathContainer{P<:Flat, T<:Real}
    n_len_tspan::Int      # these are the number of lensed field time evals computed in the path
    len_tspan::Array{T,1} # == runge_kutta_times(n_len_tspan)[1:2:end]
    rk_tspan::Array{T,1}  # == runge_kutta_times(n_len_tspan)
    # ----------
    vx::Vector{Matrix{T}} # vector field displacements which don't depend on time i.e. ∇ϕ == vx for lensing
    # ----- ↓ evaluated on time points in len_tspan
    fx::Vector{Vector{Matrix{T}}} # fx[indx_len_tspan][1] == qx_t, fx[indx_len_tspan][2] == ux_t for S2Fields
    # ----- ↓ evaluated on time points in runge_kutta_times(n_len_tspan) so n_len_tspan needs to be divisible by 2
    px::Vector{Vector{Matrix{T}}} # px[indx_rk_tspan][2] == (px_t)_{2}
    Mx::Vector{Matrix{Matrix{T}}} # Mx[indx_rk_tspan][1,2] == (invMx_t)_{1,2}
    # ---- ↓ for temp storage
    sx::Matrix{T}
    sk::Matrix{Complex{T}}

    function FlowPathContainer{P,T}(n_len_tspan, n_fields)  where {θ, nside, P<:Flat{θ,nside},T<:Real}
        rk_tspan   = T.(runge_kutta_times(n_len_tspan))
        len_tspan  = rk_tspan[1:2:end]

        vx = Vector{Matrix{T}}(undef, 2)
        vx[1] = fill(T(0),nside, nside)
        vx[2] = fill(T(0),nside, nside)
        if n_fields == 0
            fx  = Vector{Vector{Matrix{T}}}(undef, 0)
        else
            fx  = Vector{Vector{Matrix{T}}}(undef, length(len_tspan))  # fx[indx_len_tspan][1] == qx_t, fx[indx_len_tspan][2] == ux_t for S2Fields
            for i=1:length(len_tspan)
                fx[i] = Matrix{T}[fill(T(0),nside, nside) for ii ∈ 1:n_fields]
            end
        end
        px  = Vector{Vector{Matrix{T}}}(undef, length(rk_tspan))   # px[indx_rk_tspan][2] == (px_t)_{2}
        Mx  = Vector{Matrix{Matrix{T}}}(undef, length(rk_tspan))   # Mx[indx_rk_tspan][1,2] == (invMx_t)_{1,2}
        for i=1:length(rk_tspan)
            px[i] = Matrix{T}[fill(T(0),nside, nside) for ii ∈ 1:2]
            Mx[i] = Matrix{T}[fill(T(0),nside, nside) for ii ∈ 1:2, jj ∈ 1:2]
        end
        sk  = fill(Complex{T}(0), nside÷2+1, nside)
        sx  = fill(T(0),nside, nside)
        return new{P,T}(n_len_tspan, len_tspan, rk_tspan, vx, fx, px, Mx, sx, sk)
    end
end

struct Flowϕ{P<:Flat,T<:Real} <: AbstractFlow{P,T}
    ϕ::Tfourier{P,T}
    tstart::Int
    tstop::Int
    FPC0::FlowPathContainer{P,T} # empty FPC0.fx here
end

struct Flowϕᵀ{P<:Flat,T<:Real} <: AbstractFlow{P,T}
    ϕ::Tfourier{P,T}
    tstart::Int
    tstop::Int
    FPC0::FlowPathContainer{P,T} # empty FPC0.fx here
end




#################################

# constructors for Flowϕ and Flowϕω and their transposes

################################

function Flowϕ{P,T}(ϕ::Field{P,T,S0}, tstart::Int = 0, tstop::Int = 1, n_len_tspan::Int = default_ode_steps) where {nside, θ, P<:Flat{θ,nside}, T<:Real}
    FPC0 = FlowPathContainer{P,T}(n_len_tspan, 0)
    add_vx!(FPC0, ϕ)
    add_Mx_px!(FPC0, ϕ)
    return Flowϕ{P,T}(ϕ, tstart, tstop, FPC0)
end

function Flowϕᵀ{P,T}(ϕ::Field{P,T,S0}, tstart::Int = 0, tstop::Int = 1, n_len_tspan::Int = default_ode_steps) where {nside, θ, P<:Flat{θ,nside}, T<:Real}
    FPC0 = FlowPathContainer{P,T}(n_len_tspan, 0)
    add_vx!(FPC0, ϕ)
    add_Mx_px!(FPC0, ϕ)
    return Flowϕᵀ{P,T}(ϕ, tstart, tstop, FPC0)
end


#################################

# Make the Flow objects callable

################################

(L::Flowϕ{P,T})(f::F)  where {P<:Flat, T<:Real, F<:Field{P,T}} = flow(L, f)

(Lᵀ::Flowϕᵀ{P,T})(f::F)  where {P<:Flat, T<:Real, F<:Field{P,T}} = flow(L, f)


##########################################

# additional transformation constructors

###############################################


function adjoint(L::Flowϕ{P,T}) where {P<:Flat, T<:Real}
    return Flowϕᵀ{P,T}(L.ϕ, L.tstop, L.tstart, L.FPC0)
end

function adjoint(L::Flowϕᵀ{P,T}) where {P<:Flat, T<:Real}
    return Flowϕ{P,T}(L.ϕ, L.tstop, L.tstart, L.FPC0)
end



#################################

# impliment `inv` method (as required by AbstractFlow)

################################


function inv(L::Flowϕ{P,T}) where {P<:Flat, T<:Real}
    return Flowϕ{P,T}(L.ϕ, L.tstop, L.tstart, L.FPC0)
end

function inv(L::Flowϕᵀ{P,T}) where {P<:Flat, T<:Real}
    return Flowϕᵀ{P,T}(L.ϕ, L.tstop, L.tstart, L.FPC0)
end




#################################

# impliment `flow`  method (as required by AbstractFlow)

################################
# way too much code duplication here ...


@inbounds function flow(L::Flowϕ{P,T}, f::F) where {θ, nside, P<:Flat{θ,nside}, T<:Real, F<:Field{P,T}}
    if L.tstart == L.tstop
        return F(f)
    else
        fx = Matrix{T}[copy(ff) for ff in data(LenMapBasis(f))]
        n_fields = length(fx)
        𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Matrix{T}[zeros(T,nside,nside) for i∈1:n_fields], 5)
        write_op! = Write_xk_op!(P,T)

        rk_tspan, ϵ = runge_kutta_times(L.FPC0.n_len_tspan, L.tstart, L.tstop, T)
        len_tspan = rk_tspan[1:2:end]

        ϵ_by_2, ϵ_by_3, ϵ_by_6 = ϵ/2, ϵ/3, ϵ/6
        for i = 1:length(len_tspan)-1
            t = len_tspan[i]

            I_rk = find_t_in_tspan(t, L.FPC0.rk_tspan)
            update_𝒱_Flow!(𝒱_fx1, L.FPC0.px[I_rk], fx, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx1[j] .* ϵ_by_2
            end

            I_rk = find_t_in_tspan(t + ϵ_by_2, L.FPC0.rk_tspan)
            update_𝒱_Flow!(𝒱_fx2, L.FPC0.px[I_rk], fx_tmp, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx2[j] .* ϵ_by_2
            end

            # I_rk = find_t_in_tspan(t + ϵ_by_2, L.FPC0.rk_tspan)
            update_𝒱_Flow!(𝒱_fx3, L.FPC0.px[I_rk], fx_tmp, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx3[j] .* ϵ
            end

            I_rk = find_t_in_tspan(t + ϵ, L.FPC0.rk_tspan)
            update_𝒱_Flow!(𝒱_fx4, L.FPC0.px[I_rk], fx_tmp, write_op!)

            for j = 1:n_fields
                #fx[j] .+= ϵ .* (𝒱_fx1[j] .+ 2 .* 𝒱_fx2[j] .+ 2 .* 𝒱_fx3[j] .+ 𝒱_fx4[j]) ./ 6
                fx[j] .+= (𝒱_fx1[j].+𝒱_fx4[j]).*ϵ_by_6 .+ (𝒱_fx2[j].+𝒱_fx3[j]).*ϵ_by_3 
                #broadcast!((f,ϵ,v₁,v₂,v₃,v₄)->(f+ϵ*(v₁+2v₂+2v₃+v₄)/6), fx[j], (fx[j],ϵ,𝒱_fx1[j],𝒱_fx2[j],𝒱_fx3[j],𝒱_fx4[j])...)
            end
        end
        return F(LenMapBasis(P,T,fx...))
    end
end


function flow(L::Flowϕᵀ{P,T}, f::F) where {θ, nside, P<:Flat{θ,nside}, T<:Real, F<:Field{P,T}}
    if L.tstart == L.tstop
        return F(f)
    else
        fx = Matrix{T}[copy(ff) for ff in data(LenMapBasis(f))]
        n_fields = length(fx)
        𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Matrix{T}[zeros(T,nside,nside) for i∈1:n_fields], 5)
        write_op! = Write_xk_op!(P,T)

        rk_tspan, ϵ = runge_kutta_times(L.FPC0.n_len_tspan, L.tstart, L.tstop, T)
        len_tspan = rk_tspan[1:2:end]

        ϵ_by_2, ϵ_by_3, ϵ_by_6 = ϵ/2, ϵ/3, ϵ/6
        for i = 1:length(len_tspan)-1
            t = len_tspan[i]

            I_rk = find_t_in_tspan(t, L.FPC0.rk_tspan)
            update_𝒱_Flowᵀ!(𝒱_fx1, L.FPC0.px[I_rk], fx, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx1[j] .* ϵ_by_2
            end

            I_rk = find_t_in_tspan(t + ϵ_by_2, L.FPC0.rk_tspan)
            update_𝒱_Flowᵀ!(𝒱_fx2, L.FPC0.px[I_rk], fx_tmp, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx2[j] .* ϵ_by_2
            end

            # I_rk = find_t_in_tspan(t + ϵ_by_2, L.FPC0.rk_tspan)
            update_𝒱_Flowᵀ!(𝒱_fx3, L.FPC0.px[I_rk], fx_tmp, write_op!)
            for j = 1:n_fields
                fx_tmp[j] .= fx[j] .+ 𝒱_fx3[j] .* ϵ
            end

            I_rk = find_t_in_tspan(t + ϵ, L.FPC0.rk_tspan)
            update_𝒱_Flowᵀ!(𝒱_fx4, L.FPC0.px[I_rk], fx_tmp, write_op!)

            for j = 1:n_fields
                #fx[j] .+= ϵ .* (𝒱_fx1[j] .+ 2 .* 𝒱_fx2[j] .+ 2 .* 𝒱_fx3[j] .+ 𝒱_fx4[j]) ./ 6
                fx[j] .+=  (𝒱_fx1[j] .+ 𝒱_fx4[j]) .* ϵ_by_6 .+  (𝒱_fx2[j]  .+ 𝒱_fx3[j]) .* ϵ_by_3
            end
        end
        return F(LenMapBasis(P,T,fx...))
    end
end



#################################

# some flows as above but with time path cache

################################


@inbounds function add_fx!(FPC::FlowPathContainer{P,T}, f0::Field{P,T}) where {θ, nside, P<:Flat{θ,nside}, T<:Real}
    f0x = Matrix{T}[copy(ff) for ff in data(LenMapBasis(f0))]
    n_fields = length(f0x)
    𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Matrix{T}[zeros(T,nside,nside) for i∈1:n_fields], 5)
    write_op! = Write_xk_op!(P,T)

    ϵ = FPC.len_tspan[2] - FPC.len_tspan[1]

    for j = 1:n_fields
        FPC.fx[1][j] .= f0x[j]
    end

    ϵ_by_2, ϵ_by_3, ϵ_by_6 = ϵ/2, ϵ/3, ϵ/6
    for i = 1:length(FPC.len_tspan)-1
        t = FPC.len_tspan[i]

        I_rk  = find_t_in_tspan(t, FPC.rk_tspan)
        update_𝒱_Flow!(𝒱_fx1, FPC.px[I_rk], FPC.fx[i], write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx1[j] .* ϵ_by_2
        end

        I_rk  = find_t_in_tspan(t+ϵ_by_2, FPC.rk_tspan)
        update_𝒱_Flow!(𝒱_fx2, FPC.px[I_rk], fx_tmp, write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx2[j] .* ϵ_by_2
        end

        # I_rk  = find_t_in_tspan(t+ϵ_by_2, FPC.rk_tspan)
        update_𝒱_Flow!(𝒱_fx3, FPC.px[I_rk], fx_tmp, write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx3[j] .* ϵ
        end

        I_rk  = find_t_in_tspan(t+ϵ, FPC.rk_tspan)
        update_𝒱_Flow!(𝒱_fx4, FPC.px[I_rk], fx_tmp, write_op!)

        for j = 1:n_fields
            FPC.fx[i+1][j] .= FPC.fx[i][j] .+ (𝒱_fx1[j] .+ 𝒱_fx4[j]).*ϵ_by_6 .+ (𝒱_fx2[j] .+ 𝒱_fx3[j]).*ϵ_by_3
        end

    end

    return nothing
end


@inbounds function add_fxᵀ!(FPC::FlowPathContainer{P,T}, f0::Field{P,T}) where {θ, nside, P<:Flat{θ,nside}, T<:Real}
    f0x = Matrix{T}[copy(ff) for ff in data(LenMapBasis(f0))]
    n_fields = length(f0x)
    𝒱_fx1, 𝒱_fx2, 𝒱_fx3, 𝒱_fx4, fx_tmp  = @repeated(Matrix{T}[zeros(T,nside,nside) for i∈1:n_fields], 5)
    write_op! = Write_xk_op!(P,T)

    ϵ = FPC.len_tspan[2] - FPC.len_tspan[1]

    for j = 1:n_fields
        FPC.fx[1][j] .= f0x[j]
    end

    ϵ_by_2, ϵ_by_3, ϵ_by_6 = ϵ/2, ϵ/3, ϵ/6
    for i = 1:length(FPC.len_tspan)-1
        t = FPC.len_tspan[i]

        I_rk  = find_t_in_tspan(t, FPC.rk_tspan)
        update_𝒱_Flowᵀ!(𝒱_fx1, FPC.px[I_rk], FPC.fx[i], write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx1[j] .* ϵ_by_2
        end

        I_rk  = find_t_in_tspan(t+ϵ_by_2, FPC.rk_tspan)
        update_𝒱_Flowᵀ!(𝒱_fx2, FPC.px[I_rk], fx_tmp, write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx2[j] .* ϵ_by_2
        end

        # I_rk  = find_t_in_tspan(t+ϵ_by_2, FPC.rk_tspan)
        update_𝒱_Flowᵀ!(𝒱_fx3, FPC.px[I_rk], fx_tmp, write_op!)
        for j = 1:n_fields
            fx_tmp[j] .= FPC.fx[i][j] .+ 𝒱_fx3[j] .* ϵ
        end

        I_rk  = find_t_in_tspan(t+ϵ, FPC.rk_tspan)
        update_𝒱_Flowᵀ!(𝒱_fx4, FPC.px[I_rk], fx_tmp, write_op!)

        for j = 1:n_fields
            FPC.fx[i+1][j] .= FPC.fx[i][j] .+ (𝒱_fx1[j] .+ 𝒱_fx4[j]).*ϵ_by_6 .+ (𝒱_fx2[j] .+ 𝒱_fx3[j]).*ϵ_by_3
        end

    end

    return nothing
end




#################################

# ODE velocities

################################

# Flowϕ velocity
@inbounds function update_𝒱_Flow!(𝒱_fx::A, px::A, fx::A, write_op!::Write_xk_op!{T,F}) where {T,F,A<:Vector{Matrix{T}}}
    for f ∈ 1:length(fx)
        write_op!(𝒱_fx[f], (px[1], px[2]), (1,2), fx[f]) # pxⁱ ⋅ ∇ⁱ⋅ fxᶠ  ⟶ outxᶠ
    end
    return nothing
end

# Flowϕᵀ velocity
@inbounds function update_𝒱_Flowᵀ!(𝒱_fx::A, px::A, fx::A, write_op!::Write_xk_op!{T,F}) where {T,F,A<:Vector{Matrix{T}}}
    for f ∈ 1:length(fx)
        write_op!(𝒱_fx[f], (1,2), (px[1], px[2]), fx[f]) # ∇ⁱ ⋅ pxⁱ ⋅ fxᶠ  ⟶ outxᶠ
    end
    return nothing
end



######################################################
#
# struct for FlowPathContainer with  methods to be used in conjunction with constructors
#
######################################################

@inbounds function add_vx!(FPC::FlowPathContainer{P,T}, ϕ::Field{P,T,S0}) where {θ,nside, P<:Flat{θ,nside}, T<:Real}
    g = r𝔽(P,T)
    ϕk, = ϕ |> Tfourier{P,T} |> data

    FPC.sk .=  im .* g.k[1] .* ϕk
    ldiv!(FPC.vx[1], g.FFT, FPC.sk)

    FPC.sk .=  im .* g.k[2] .* ϕk
    ldiv!(FPC.vx[2], g.FFT, FPC.sk)

    return nothing
end


@inbounds function add_Mx_px!(FPC::FlowPathContainer{P,T}, ϕ::Field{P,T,S0}) where {θ, nside, P<:Flat{θ,nside}, T<:Real}
    g = r𝔽(P,T)
    ϕk, = ϕ |> Tfourier{P,T} |> data
    ∂1_v1x = g \ ( .- g.k[1] .* g.k[1] .* ϕk)
    ∂1_v2x = g \ ( .- g.k[1] .* g.k[2] .* ϕk)
    ∂2_v2x = g \ ( .- g.k[2] .* g.k[2] .* ϕk)

    detM = Vector{Matrix{T}}(undef, length(FPC.rk_tspan))
    Base.Threads.@threads for i in 1:length(FPC.rk_tspan)
        detM[i] = fill(T(0), nside, nside)
        for cm = 1:nside
            for rw = 1:nside
                FPC.Mx[i][1,1][rw,cm]  = 1 + FPC.rk_tspan[i] * ∂2_v2x[rw,cm]
                FPC.Mx[i][1,2][rw,cm]  =   - FPC.rk_tspan[i] * ∂1_v2x[rw,cm]
                FPC.Mx[i][2,2][rw,cm]  = 1 + FPC.rk_tspan[i] * ∂1_v1x[rw,cm]
                detM[i][rw,cm]         = FPC.Mx[i][1,1][rw,cm] * FPC.Mx[i][2,2][rw,cm] - FPC.Mx[i][1,2][rw,cm]^2
                FPC.Mx[i][1,1][rw,cm]  = FPC.Mx[i][1,1][rw,cm]/detM[i][rw,cm]
                FPC.Mx[i][1,2][rw,cm]  = FPC.Mx[i][1,2][rw,cm]/detM[i][rw,cm]
                FPC.Mx[i][2,2][rw,cm]  = FPC.Mx[i][2,2][rw,cm]/detM[i][rw,cm]
                FPC.Mx[i][2,1][rw,cm]  = FPC.Mx[i][1,2][rw,cm]

                FPC.px[i][1][rw,cm] = FPC.Mx[i][1,1][rw,cm] * FPC.vx[1][rw,cm] + FPC.Mx[i][1,2][rw,cm] * FPC.vx[2][rw,cm]
                FPC.px[i][2][rw,cm] = FPC.Mx[i][2,1][rw,cm] * FPC.vx[1][rw,cm] + FPC.Mx[i][2,2][rw,cm] * FPC.vx[2][rw,cm]
            end
        end
    end
    return nothing
end





###################################################
#
# printing
#
#################################################

# Flowϕ{P,T}
function show(io::IO, L::Flowϕ{P,T}) where {T<:Real, P <: Flat{Θpix,n}} where {Θpix,n}
    lng_digit_print = 2
    pround = round(Θpix; sigdigits = lng_digit_print)
    println(io, "Flowϕ{P,T} from time $(L.tstart) → $(L.tstop) where P = Flat{$(pround),$n} and T = $T")
end


# Flowϕᵀ{P,T}
function show(io::IO, L::Flowϕᵀ{P,T}) where {T<:Real, P <: Flat{Θpix,n}} where {Θpix,n}
    lng_digit_print = 2
    pround = round(Θpix; sigdigits = lng_digit_print)
    println(io, "Flowϕᵀ{P,T} from time $(L.tstart) → $(L.tstop) where P = Flat{$(pround),$n} and T = $T")
end
