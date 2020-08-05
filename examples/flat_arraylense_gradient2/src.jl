

using FieldLensing
using Test
using Spectra
using XFields
using FFTransforms
using LBblocks
using SparseArrays
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using LoopVectorization
using PyPlot
using NLopt


# To use ArrayLense we just need to define ∇!
# -----------------------------------------------
struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,B}) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end
function (∇!::Nabla!{Tθ,Tφ})(y::NTuple{2,B}) where {Tθ,Tφ,Tf,B<:AbstractMatrix{Tf}}
    ∇y = (similar(y[1]), similar(y[2]))
    ∇!(∇y, (y[1],y[2]))
    ∇y
end


function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::B) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    ∇!(∇y, (y,y))
end
function (∇!::Nabla!{Tθ,Tφ})(y::B) where {Tθ,Tφ,Tf,B<:AbstractMatrix{Tf}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end


function whitemap(trm::T) where T<:Transform
    zx = randn(eltype_in(trm),size_in(trm))
    Xmap(trm, zx ./ √Ωx(trm))
end


# custom pcg with function composition (Minv * A \approx I)
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol::Float64 = 1e-8)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = XFields.nan2zero(sqrt(dot(r,r)/dot(b,b)))
        if rel_error < rel_tol
            return x, reshist
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x, reshist
end


LinearAlgebra.dot(f::Xfield,g::Xfield) = Ωx(fieldtransform(f)) * dot(f[:],g[:])



# set the transform and the gradient operator 
# -----------------------------------------------
trm, ∇! = @sblock let Δθ′ = 2.5, Δφ′ = 2.5, nθ = 512, nφ = 512
	## 𝕨      = r𝕎32(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
	𝕨      = r𝕎(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
	trm    = ordinary_scale(𝕨)*𝕨

	onesnθm1 = fill(1,nθ-1)
	∂θ = spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
	∂θ[1,end] = -1
    ∂θ[end,1] =  1
    ∂θ = (1 / (2 * Δpix(trm)[1])) * ∂θ

    onesnφm1 = fill(1,nφ-1)
    ∂φ      = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
    ## for the periodic boundary conditions
    ∂φ[1,end] = -1
    ∂φ[end,1] =  1
    ## now as a right operator
    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
    ∂φᵀ = transpose((1 / (2*Δpix(trm)[2])) * ∂φ);

    ∇! = Nabla!(∂θ, ∂φᵀ)

    return trm, ∇!
end


# ------------------------
Cn, Cf, Cϕ, Δ = @sblock let trm
	l   = wavenum(trm)
    
    μKarcminT = 5
    cnl = deg2rad(μKarcminT/60)^2  .+ 0 .* l
	Cn  = DiagOp(Xfourier(trm, cnl)) 

    cfl = Spectra.cTl_besselj_approx.(l)
	Cf  = DiagOp(Xfourier(trm, cfl))

    scale_ϕ = 1.5 
	cϕl     = scale_ϕ .* Spectra.cϕl_approx.(l) 
    Cϕ      = DiagOp(Xfourier(trm, cϕl))

    Δ       = DiagOp(Xfourier(trm, .- l .^ 2))

    Cn, Cf, Cϕ, Δ
end;


#-

n, f, ϕ, Ł = @sblock let trm, ∇!, Cn, Cf, Cϕ
    f = √Cf * whitemap(trm)
    n = √Cn * whitemap(trm)
    ϕ = √Cϕ * whitemap(trm)
    Ł = ϕ -> FieldLensing.ArrayLense(∇!(ϕ[:]), ∇!, 0, 1, 16)
    
    MapField(n), MapField(f), MapField(ϕ), Ł
end;

#-

d = Ł(ϕ) * f + n


# Gradient update and WF closure 
# ------------------------------------------

update_ϕ, update_f₀f₁, ϕf₁_logP, ∇ϕf₁_logP, ∇ϕf₀_logP, ∇f₀_logP = @sblock let trm, d, Cf, Cϕ, Cn, Δ, ∇!, Ł

    # useful for testing
    ∇f₀_logP = (ϕ₀,f₀) -> Ł(ϕ₀)' / Cn * (d - Ł(ϕ₀) * f₀) - Cf \ f₀

    ∇ϕf₀_logP = function (ϕ,f₀) 
        f₁       = Ł(ϕ)*f₀
        v        = ∇!(ϕ[:])
        τŁ₁₀     = FieldLensing.τArrayLense(v, (f₁[:],), ∇!, 1, 0, 16)        
        τv₀, τf₀ = τŁ₁₀ * (map(zero,v),  ((Cn\(d-f₁))[:],) )
        ∇ϕ       = Xmap(trm, - sum(∇!(τv₀))) - Cϕ \ ϕ
        ∇f₀      = Xmap(trm, τf₀[1]) - Cf \ f₀
        return ∇ϕ, ∇f₀
    end

    ∇ϕf₁_logP = function (ϕ,f₁)
        f₀       = Ł(ϕ)\f₁
        v        = ∇!(ϕ[:])
        τŁ₀₁     = FieldLensing.τArrayLense(v, (f₀[:],), ∇!, 0, 1, 16)
        ∇ϕf₀     = ∇ϕf₀_logP(ϕ,f₀)       
        τv₁, τf₁ = τŁ₀₁ * (∇!(∇ϕf₀[1][:]),  (∇ϕf₀[2][:],) )
        ∇ϕ       = Xmap(trm, - sum(∇!(τv₁)))
        ∇f₁      = Xmap(trm, τf₁[1])
        return ∇ϕ, ∇f₁
    end

    ϕf₁_logP = function (ϕ,f₁)
        # rtn  = d-f₁    |> x -> - dot(x, Cn \ x) / 2
        rtn1 = sqrt(Cf) \ (Ł(ϕ)\f₁) |> x -> - dot(x, x) / 2
        rtn2  = sqrt(Cϕ) \ ϕ        |> x -> - dot(x, x) / 2
        return rtn1 + rtn2 
    end

    update_f₀f₁ = function (ϕ,f₁)
        f₀  = Ł(ϕ)\f₁
        simf₀, wfhist = pcg(
            f -> inv(inv(Cf) + inv(Cn)) * f, 
            f -> Ł(ϕ)' * inv(Cn) * Ł(ϕ) * f +  Cf \ f,
            Ł(ϕ)' * inv(Cn) * (d + √Cn * whitemap(trm)) + Cf \ (√Cf * whitemap(trm)),
            nsteps = 50,
            rel_tol = 1e-20,
        )
        simf₁ = Ł(ϕ) * f₀
        return simf₀, simf₁, wfhist
    end


    update_ϕ = function (ϕ,f₁)

        # solver=:LN_SBPLX # :LN_SBPLX, :LN_COBYLA, :LN_NELDERMEAD, :GN_DIRECT_L, :GN_DIRECT_L_RAND
        solver=:LN_COBYLA 

        Cnϕ = 0.1 * maximum(real.( (Δ * Δ * Cϕ)[!] )) * inv(Δ^2)
        invΛ∇vcurr = (inv(Cnϕ) + inv(Cϕ)) \ ∇ϕf₁_logP(ϕ,f₁)[1]
        #invΛ∇vcurr = Cϕ * ∇ϕf₁_logP(ϕ,f₁)[1]
        
        T = eltype_in(trm)
        opt = NLopt.Opt(solver, 1)
        opt.maxtime      = 30
        opt.upper_bounds = T[1.5]
        opt.lower_bounds = T[0]
        opt.initial_step = T[0.00001]
        opt.max_objective = function (β, grad)
            ϕβ = ϕ+β[1]*invΛ∇vcurr
            ϕf₁_logP(ϕ+β[1]*invΛ∇vcurr, f₁)
        end

        ll_opt, β_opt, = NLopt.optimize(opt,  T[0.000001])
        @show ll_opt, β_opt
        
        return ϕ + β_opt[1] * invΛ∇vcurr
    end


    return update_ϕ, update_f₀f₁, ϕf₁_logP, ∇ϕf₁_logP, ∇ϕf₀_logP, ∇f₀_logP
end 


  
f₁curr = d 
ϕcurr  = 0*d

f₀curr, f₁curr, wfhist = update_f₀f₁(ϕcurr, f₁curr);
ϕcurr = update_ϕ(ϕcurr,f₁curr);

# first test (t1 and t2 should be the same) 
t1 = ∇ϕf₀_logP(ϕ, f₀curr)[2]
t2 = ∇f₀_logP(ϕ, f₀curr)

t1[:] .- t2[:] |> matshow; colorbar()
t2[:] |> matshow; colorbar()
# ✓


# ----------------

Cnϕ = 0.2 * maximum(real.( (Δ * Δ * Cϕ)[!] )) * inv(Δ^2)
nH = inv(inv(Cnϕ) + inv(Cϕ))
# (Δ^2 * nH)[!][:,1] |> loglog
# (Δ^2 * Cϕ)[!][:,1] |> loglog

∇ϕ, ∇f₁ = ∇ϕf₁_logP(ϕcurr, f₀curr)
(nH*∇ϕ)[:] |> matshow
ϕ[:] |> matshow



 semilogy(wfhist)

ϕcurr[:] |> matshow; colorbar()
ϕ[:] |> matshow; colorbar()



∇ϕf₀_logP = function (ϕ,f₀) 
    f₁       = Ł(ϕ)*f₀
    v        = ∇!(ϕ[:])
    τŁ₁₀     = FieldLensing.τArrayLense(v, (f₁[:],), ∇!, 1, 0, 16)        
    τv₀, τf₀ = τŁ₁₀ * (map(zero,v),  ((Cn\(d-Ł(ϕ)*f₀))[:],) )
    ∇ϕ       = Xmap(trm, - sum(∇!(τv₀))) - Cϕ \ ϕ
    ∇f₀      = Xmap(trm, τf₀[1]) - Cf \ f₀
    return ∇ϕ, ∇f₀
end

∇ϕf₁_logP = function (ϕ,f₁)
    f₀       = Ł(ϕ)\f₁
    v        = ∇!(ϕ[:])
    τŁ₀₁     = FieldLensing.τArrayLense(v, (f₀[:],), ∇!, 0, 1, 16)
    ∇ϕf₀     = ∇ϕf₀_logP(ϕ,f₀)       
    τv₁, τf₁ = τŁ₀₁ * (∇!(∇ϕf₀[1][:]),  (∇ϕf₀[2][:],) )
    ∇ϕ       = Xmap(trm, - sum(∇!(τv₁)))
    ∇f₁      = Xmap(trm, τf₁[1])
    return ∇ϕ, ∇f₁
end

