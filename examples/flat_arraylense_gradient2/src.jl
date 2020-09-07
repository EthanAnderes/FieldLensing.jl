
using FFTW
FFTW.set_num_threads(5)

using FieldLensing
using FieldLensing: ArrayLense, ArrayLenseᴴ, Gradient

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
using Test


# To use ArrayLense we just need to define ∇! which isa Gradient{m}
# where m is the number of partial derivatives
# -----------------------------------------------

struct Nabla!{Tθ,Tφ} <: FieldLensing.Gradient{2}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function LinearAlgebra.adjoint(∇!::Nabla!)
    return Nabla!(
        ∇!.∂θ',
        ∇!.∂φᵀ',
    )
end

function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{1}) where {Tθ,Tφ} 
    mul!(des, ∇!.∂θ, y)
end

function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{2}) where {Tθ,Tφ}
    mul!(des, y, ∇!.∂φᵀ)
end 


# Other methods 
# -------------------------------------


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

    θℝ=pix(trm)[1]
    Δθℝ = θℝ[2] - θℝ[1]
    ∂θ′ = spdiagm(
            -2 => fill( 1,length(θℝ)-2),
            -1 => fill(-8,length(θℝ)-1),
             1 => fill( 8,length(θℝ)-1),
             2 => fill(-1,length(θℝ)-2),
            )
    ∂θ′[1,end]   =  -8
    ∂θ′[1,end-1] =  1
    ∂θ′[2,end]   =  1

    ∂θ′[end,1]   =  8
    ∂θ′[end,2]   = -1
    ∂θ′[end-1,1] = -1

    ∂θ = (1 / (12Δθℝ)) * ∂θ′
    ## ∂θ = (∂θ - ∂θ') / 2 # not needed


    φℝ=pix(trm)[2]
    Δφℝ = φℝ[2] - φℝ[1]
    ∂φ  = spdiagm(
            -2 => fill( 1,length(φℝ)-2),
            -1 => fill(-8,length(φℝ)-1),
             1 => fill( 8,length(φℝ)-1),
             2 => fill(-1,length(φℝ)-2),
            )
    ∂φ[1,end]   =  -8
    ∂φ[1,end-1] =  1
    ∂φ[2,end]   =  1
    ∂φ[end,1]   =  8
    ∂φ[end,2]   =  -1
    ∂φ[end-1,1] =  -1
    ∂φᵀ = transpose((1 / (12Δφℝ)) * ∂φ)
    ## ∂φᵀ = (∂φᵀ - ∂φᵀ') / 2 # not needed

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
    Ł = x -> FieldLensing.ArrayLense(∇!(x[:]), ∇!, 0, 1, 16)
    
    MapField(n), MapField(f), MapField(ϕ), Ł
end;

#-

d = Ł(ϕ) * f + n

ds = (;trm, d, Cf, Cϕ, Cn, Δ, ∇!, Ł)

#=
v     = ∇!(ϕ[:])
τŁ₁₀  = FieldLensing.τArrayLense(v, (f[:],), ∇!, 1, 0, 16)        
τv, τf = map(zero,v),  ((Cn\(d-f))[:],)
τϕ      = zero(v[1])
τv₀, τf₀′ = τŁ₁₀ * (τv, τf)
τϕ₀, τf₀ = FieldLensing.τpotential(τŁ₁₀, τϕ, τf)
@benchmark $τŁ₁₀ * $((τv, τf))
@benchmark FieldLensing.τpotential($τŁ₁₀, $τϕ, $τf)
=#


# function ∇ϕf₀_logP(ϕ, f₀, ds) 
#     v, f₁  = ds.∇!(ϕ[:]), ds.Ł(ϕ) * f₀
#     τŁ₁₀   = FieldLensing.τArrayLense(v, (f₁[:],), ds.∇!, 1, 0, 16)        
#     τϕ, τf = FieldLensing.τpotential(τŁ₁₀, zero(ϕ[:]),  ((ds.Cn\(ds.d-f₁))[:],) )
#     ∇ϕ     = Xmap(ds.trm, τϕ) - ds.Cϕ \ ϕ
#     ∇f     = Xmap(ds.trm, τf[1]) - ds.Cf \ f₀
#     return ∇ϕ, ∇f
# end
# function ∇ϕf₁_logP(ϕ,f₁, ds) 
#     v, f₀    = ds.∇!(ϕ[:]), ds.Ł(ϕ) \ f₁
#     τŁ₀₁     = FieldLensing.τArrayLense(v, (f₀[:],), ds.∇!, 0, 1, 16)
#     ∇ϕ₀, ∇f₀ = ∇ϕf₀_logP(ϕ,f₀, ds)       
#     τϕ, τf   = FieldLensing.τpotential(τŁ₀₁, ∇ϕ₀[:],  (∇f₀[:],))
#     ∇ϕ₁      = Xmap(ds.trm, τϕ)
#     ∇f₁      = Xmap(ds.trm, τf[1])
#     return ∇ϕ₁, ∇f₁
# end


# this one is much faster and doesn't need the potential flow ...
function ∇ϕf₁_logP(ϕ, f₁, ds) 

    v, f₀  = ds.∇!(ϕ[:]), ds.Ł(ϕ) \ f₁
    τŁ₀₁   = FieldLensing.τArrayLense(v, (f₀[:],), ds.∇!, 0, 1, 16)
    τŁ₁₀   = FieldLensing.τArrayLense(v, (f₁[:],), ds.∇!, 1, 0, 16)        

    τv₀, τf₀ = τŁ₁₀(map(zero,v),  ((ds.Cn\(ds.d-f₁))[:],))
    ∇f₀      = Xmap(ds.trm, τf₀[1]) - ds.Cf \ f₀

    τv₁, τf₁ = τŁ₀₁(τv₀,  (∇f₀[:],))
    ∇ϕ₁      = Xmap(ds.trm, -sum(ds.∇!(τv₁))) - ds.Cϕ \ ϕ
    ∇f₁      = Xmap(ds.trm, τf₁[1])

    return ∇ϕ₁, ∇f₁
end

function ϕf₁_logP(ϕ,f₁, ds) 
    rtn1  = sqrt(ds.Cn) \ (ds.d-f₁)    |> x -> - dot(x, x) / 2
    rtn2  = sqrt(ds.Cf) \ (ds.Ł(ϕ)\f₁) |> x -> - dot(x, x) / 2
    rtn3  = sqrt(ds.Cϕ) \ ϕ            |> x -> - dot(x, x) / 2
    return rtn1 + rtn2 + rtn3
end

function update_f₁f₀(ϕ, ds)
    simf₀, wfhist = pcg(
        f -> inv(inv(ds.Cf) + inv(ds.Cn)) * f, 
        f -> ds.Ł(ϕ)' * inv(ds.Cn) * ds.Ł(ϕ) * f +  ds.Cf \ f,
        ds.Ł(ϕ)' * inv(ds.Cn) * (ds.d + √ds.Cn * whitemap(ds.trm)) + ds.Cf \ (√ds.Cf * whitemap(ds.trm)),
        nsteps = 50,
        rel_tol = 1e-20,
    )
    simf₁ = ds.Ł(ϕ) * simf₀
    return simf₁, simf₀, wfhist
end

function update_ϕ(ϕ,f₁,ds)

    solver=:LN_SBPLX # :LN_SBPLX, :LN_COBYLA, :LN_NELDERMEAD, :GN_DIRECT_L, :GN_DIRECT_L_RAND

    Cnϕ = 0.4 * maximum(real.( (ds.Δ * ds.Δ * ds.Cϕ)[!] )) * inv(ds.Δ^2)
    nH⁻¹ = inv(inv(Cnϕ) + inv(ds.Cϕ))
    ∇ϕ, ∇f₁ = ∇ϕf₁_logP(ϕ, f₁, ds)
    #(nH⁻¹*∇ϕ)[:] |> matshow; colorbar()
    nH⁻¹∇ϕ = nH⁻¹ * ∇ϕ

    T = eltype_in(ds.trm)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = 30
    opt.upper_bounds = T[1.5]
    opt.lower_bounds = T[0]
    opt.initial_step = T[0.00001]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * nH⁻¹∇ϕ
        ϕf₁_logP(ϕ + β[1] * nH⁻¹∇ϕ, f₁, ds)
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0.000001])
    @show ll_opt, β_opt
    
    return ϕ + β_opt[1] * nH⁻¹∇ϕ
end




  
ϕ₁ = 0*d

for i = 1:3
    global ϕ₁, f₁, f₀, wfhist
    f₁, f₀, wfhist = update_f₁f₀(ϕ₁, ds)
    ϕ₁ = update_ϕ(ϕ₁, f₁, ds)
end

ϕ[:] |> matshow; colorbar()
ϕ₁[:] |> matshow; colorbar()

(Δ*ϕ)[:] |> matshow; colorbar()
(Δ*ϕ₁)[:] |> matshow; colorbar()









# ---------------
test_∇ϕf₀_logP = function (ϕ, f₀, ds) 
    v, f₁  = ds.∇!(ϕ[:]), ds.Ł(ϕ) * f₀
    τŁ₁₀   = FieldLensing.τArrayLense(v, (f₁[:],), ds.∇!, 1, 0, 16)        
    τv, τf = τŁ₁₀(map(zero,v),  ((ds.Cn\(ds.d-f₁))[:],))
    ∇ϕ     = Xmap(ds.trm, -sum(ds.∇!(τv))) #  - ds.Cϕ \ ϕ
    ∇f     = Xmap(ds.trm, τf[1]) - ds.Cf \ f₀
    return ∇ϕ, ∇f
end

t1, f1 = ∇ϕf₀_logP(ϕ₁, f₀, ds)
t2, f2 = test_∇ϕf₀_logP(ϕ₁, f₀, ds)

t1[:] |> matshow; colorbar()
t2[:] |> matshow; colorbar()
t1[:] .- t2[:] |> matshow; colorbar()

f1[:] |> matshow; colorbar()
f2[:] |> matshow; colorbar()
f1[:] .- f2[:] |> matshow; colorbar()
# ✓


# ---------------
test_∇ϕf₁_logP = function (ϕ,f₁, ds) 
    v, f₀  = ds.∇!(ϕ[:]), ds.Ł(ϕ) \ f₁
    τŁ₀₁   = FieldLensing.τArrayLense(v, (f₀[:],), ds.∇!, 0, 1, 16)
    τŁ₁₀   = FieldLensing.τArrayLense(v, (f₁[:],), ds.∇!, 1, 0, 16)        

    τv₀, τf₀ = τŁ₁₀(map(zero,v),  ((ds.Cn\(ds.d-f₁))[:],))
    ∇f₀      = Xmap(ds.trm, τf₀[1]) - ds.Cf \ f₀

    τv₁, τf₁ = τŁ₀₁(τv₀,  (∇f₀[:],))
    ∇ϕ₁      = Xmap(ds.trm, -sum(ds.∇!(τv₁))) - ds.Cϕ \ ϕ
    ∇f₁      = Xmap(ds.trm, τf₁[1])

    return ∇ϕ₁, ∇f₁
end


@time t1, f1 = ∇ϕf₁_logP(ϕ₁, f₁, ds)
@time t2, f2 = test_∇ϕf₁_logP(ϕ₁, f₁, ds)

(Cϕ*t1)[:] |> matshow; colorbar()
(Cϕ*t2)[:] |> matshow; colorbar()
(Cϕ*(t1 - t2))[:] |> matshow; colorbar()

f1[:] |> matshow; colorbar()
f2[:] |> matshow; colorbar()
f1[:] .- f2[:] |> matshow; colorbar()
# ✓



# ✓




# ---------------
# test (t1 and t2 should be the same)
t1, t2 = @sblock let f =  f₁, ϕ = ϕ₁, ds
    t1 = ∇ϕf₀_logP(ϕ, f, ds)[2] + ds.Cf \ f
    t2 = ds.Ł(ϕ)' * (ds.Cn \ (ds.d - ds.Ł(ϕ) * f))
    t1, t2
end
t1[:] |> matshow; colorbar()
t2[:] |> matshow; colorbar()
t1[:] .- t2[:] |> matshow; colorbar()

# ✓

