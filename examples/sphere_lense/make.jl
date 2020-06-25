#src Build with 
#src  ```
#src  julia make.jl
#src  jupyter nbconvert note.ipynb
#src  ```
using Literate                      #src
config = Dict(                      #src
    "documenter"    => false,       #src
    "execute"       => true,        #src
    "name"          => "example",   #src
    "credit"        => false,       #src
)                                   #src
Literate.notebook(                  #src
    "make.jl",                      #src
    config=config,                  #src
)                                   #src

# Using Xlense from FieldLensing with SphereTransforms for XFields
# ==========

import FFTW
FFTW.set_num_threads(5)
import FFTransforms
using FFTransforms: r𝕎, 𝕀, ⊗, ordinary_scale

using Spectra
using XFields
using FieldLensing
using SphereTransforms

using SparseArrays
using LinearAlgebra
using LBblocks: @sblock
using PyPlot
using BenchmarkTools


# Set the Xfield transform 
# ----------

trn = @sblock let T=Float64, Nside=2048 # 1024
    spin = 0
    𝕊(T, 2*Nside, spin)
end

#-

@show Δθarcmin = Δθ′(trn)
@show Δφarcmin = Δφ′(trn);

#-

l, m, b = lm(trn)

#- 

l 

#- 

m

#-
## `b` is a BitArray corresponding to which entires of the lm array are used.
b


# Compute the spectral matrices which mimic CMB tempurature and lesing potential
# ------------------------------

Ct, Cϕ = @sblock let trn
    l, m, b   = lm(trn)

    cϕl = Spectra.cϕl_approx.(l) 
    
    cTl = Spectra.cTl_approx.(l)
    #cTl = Spectra.cTl_besselj_approx.(l)
    #cTl = Spectra.cTl_matern_cov_approx.(l)
    cTl .*= Spectra.knee.(l; ell=50, alpha=2)
    cTl[l .< 2] .= 0
    cTl[.!b]    .= 0

    cϕl[l .< 2] .= 0
    cϕl[.!b]    .= 0

    Ct  = DiagOp(Xfourier(trn, cTl)) 
    Cϕ  = DiagOp(Xfourier(trn, cϕl)) 

    Ct, Cϕ
end;

# Simulate T and ϕ fields
# ---------------

T, ϕ = @sblock let trn, Ct, Cϕ
    zTlm = white_noise_lm(trn)
    zϕlm = white_noise_lm(trn)

    T = √Ct * Xfourier(trn, zTlm)
    ϕ = √Cϕ * Xfourier(trn, zϕlm)

    T, ϕ
end;

#- 
T[:] |> matshow; colorbar(); gcf()

#-
ϕ[:] |> matshow; colorbar(); gcf()



# Map derivaties w.r.t θ, φ
# --------------------
using SphereTransforms.FastTransforms: chebyshevpoints
θ = acos.(chebyshevpoints(Float64, trn.nθ; kind=1))
sin⁻¹θ = 1 ./ sin.(θ)

# nφ = trn.nφ
# nθ = trn.nθ
# IW = 𝕀(trn.nθ)⊗r𝕎(trn.nφ, 2π)
# let IW = IW , ikφ = im * (0:trn.nφ÷2)', sin⁻¹θ = inv.(sin.(acos.(chebyshevpoints(Float64, nθ; kind=1)))) # i.e. has length nφ÷2+1
#     global function ∂φ(fmap::T)::T where {R,T<:Array{R,2}}
#         planIW = FFTransforms.plan(IW)
#         return sin⁻¹θ .* (planIW \ (ikφ .* (planIW * fmap)))
#     end
#     global function ∂φ!(outmap, fmap, sk) # sk is fourier storage
#         planIW = FFTransforms.plan(IW)
#         mul!(sk, planIW.unscaled_forward_transform, fmap)
#         @inbounds @. sk = sk * ikφ * planIW.scale_forward * planIW.scale_inverse
#         mul!(outmap, planIW.unscaled_inverse_transform, sk)
#     end
# end

#-

hθ = Δθ(trn) 
# ∂θmat = spdiagm(
#         -2 => fill( 1/12/hθ, trn.nθ-2),
#         -1 => fill(-2/3/hθ,  trn.nθ-1),
#         0  => fill( 0.0,     trn.nθ),
#         1  => fill( 2/3/hθ,  trn.nθ-1),
#         2  => fill(-1/12/hθ, trn.nθ-2)  
# )
∂θmat = spdiagm(
        -1 => fill(-1/(2hθ),  trn.nθ-1),
        1  => fill( 1/(2hθ),  trn.nθ-1),
)

# let ∂θmat=∂θmat
#     global ∂θ(fmap) = ∂θmat * fmap
#     #global ∂θ!(outmap, fmap) = mul!(outmap, ∂θmat, fmap)
# end
@eval ∂θ(fmap) = $∂θmat * fmap

hφ = Δφ(trn)
∂φmatᵀ2 = (1/hφ) * spdiagm(
        -2 => fill( 1/12, trn.nφ-2),
        -1 => fill(-2/3,  trn.nφ-1),
        1  => fill( 2/3,  trn.nφ-1),
        2  => fill(-1/12, trn.nφ-2)  
)'
∂φmatᵀ = transpose(
        spdiagm(
            -1 => fill( -1/2,  trn.nφ-1),
            # 0  => fill( 0.0,      trn.nφ),
            1  => fill( 1/2,  trn.nφ-1),
        ) 
    )

# let ∂φmatᵀ=∂φmatᵀ, θ = acos.(chebyshevpoints(Float64, nθ; kind=1))
#     global ∂φ(fmap) = (1 ./ sin.(θ)) .* (fmap * ∂φmatᵀ)
#     global ∂φ!(outmap, fmap) = mul!(outmap, fmap, ∂φmatᵀ)
# end
@eval distinaz(t) = 2asin( sin(t)*sin($hφ/2) )
@eval ∂φforϕ(fmap) =  (fmap * $∂φmatᵀ) ./ $(distinaz.(θ)) 
@eval ∂φ(fmap) =  (fmap * $∂φmatᵀ2)

# φ = pix(trn)[2]
# ∂φ(ones(size(θ)).* φ')


# Use the map derivaties to compute the displacement fields
# --------------------


vϕ = @sblock let trn, ϕ, θ= θ
    ϕmap = ϕ[:]
    #Xmap(trn, ∂θ(ϕmap)), Xmap(trn, (1 ./ sin.(θ)).^2 .* ∂φ(ϕmap))
    Xmap(trn, ∂θ(ϕmap)), Xmap(trn, ∂φforϕ(ϕmap))
end;

#-

vϕ[1][:][20:end-20,:] |> matshow; colorbar(); gcf()

#-

vϕ[2][:][200:end-200,2:end-10] |> matshow; colorbar(); gcf()

#-

function α_θφ(θ1,φ1,θ2,φ2) 
    Δφ = φ1 - φ2
    Δθ = θ1 - θ2
    sθ1, sθ2 = sin(θ1), sin(θ2)
    # return acos(cos(θ1)*cos(θ2) + sin(θ1)*sin(θ2)*cos(Δφ))
    return 2asin(√(sin(Δθ/2)^2 + sθ1 * sθ2 * sin(Δφ/2)^2))
end 

θ, φ = pix(trn)
displacements = α_θφ.(θ,φ',θ .+ vϕ[1][:], φ' .+ vϕ[2][:]) 

displacements[3:end-3,3:end-3] |> matshow; colorbar()
deg2rad(2.7/60)
# Now define `plan` and `gradient!` to use Xlense for SphereTransforms
# --------------------

function FieldLensing.plan(
        L::Xlense{2,𝕊{Tf},Tf,Ti,2}
    ) where {Tf<:Float64, Ti<:Float64}
    szf, szi =  size_in(L.trn), size_out(L.trn)
    k     = freq(L.trn) |> k -> (Tf.(k[1]), Tf.(k[2]))
    vx    = tuple(L.v[1][:], L.v[2][:])
    
    sk  = zeros(Ti,szi[1], szi[2]÷2+1) # custom since it is storage for real slice fft
    yk  = zeros(Ti,szi)

    ∂vx = Array{Tf,2}[Array{Tf,2}(undef,szf) for r=1:2, c=1:2]
    # ∂θ!(∂vx[1,1], vx[1])
    # ∂θ!(∂vx[2,1], vx[2])
    # ∂φ!(∂vx[1,2], vx[1])
    # ∂φ!(∂vx[2,2], vx[2])
    θ = acos.(chebyshevpoints(Float64, L.trn.nθ; kind=1))
    ∂vx[1,1] .= ∂θ(vx[1])
    ∂vx[2,1] .= ∂θ(vx[2])
    ∂vx[1,2] .= ∂φ(vx[1])
    ∂vx[2,2] .= ∂φ(vx[2])

    mx  = deepcopy(∂vx)
    px  = deepcopy(vx)
    ∇y  = deepcopy(vx)

    FieldLensing.XlensePlan{2,𝕊{Tf},Tf,Ti,2}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
end

function FieldLensing.gradient!(
        ∇y::NTuple{2,Array{Tf,2}}, 
        y::Array{Tf,2}, 
        Lp::FieldLensing.XlensePlan{2,𝕊{Tf},Tf,Ti,2}
    )  where {Tf<:Float64, Ti<:Float64}
    ∇y[1] .= ∂θ(y)
    ∇y[2] .= ∂φ(y)
    #θ = acos.(chebyshevpoints(Float64, L.trn.nθ; kind=1))
    #∇y[2] .= sin.(θ) .* ∂φ(y)
end


# Construct lense 
# ---------------

# Now construct the lensing and adjoint lensing operator

L = @sblock let trn, vϕ, nsteps=16
    t₀ = 0
    t₁ = 1
    L  = FieldLensing.Xlense(trn, 0.5 .* vϕ, t₀, t₁, nsteps)
    L
end;

# Lense the field
# ---------------

# Forward lensing field

@time lenT1 = L * T
lenT1[:][2:300,2:300] |> matshow; colorbar(); gcf()
lenT1[:][1000:2000,2:300] |> matshow; colorbar(); gcf()

# Difference between lensed and un-lensed

(T - lenT1)[:][1000:end-1000,10:2000] |> matshow; colorbar(); gcf()

# Invert the lense and compare

T1 = L \ lenT1
(T - T1)[:] |> matshow; colorbar(); gcf()


# adjoint Lense the field
# ---------------

# Forward adjoint lensing field

lenʰT1 = Lʰ * T

# Difference between adjoint lensing and un-lensed

(T - lenʰT1)[:] |> matshow; colorbar(); gcf()

# Invert the lense and compare

ʰT1 = Lʰ \ lenʰT1
(T - T1)[:] |> matshow; colorbar(); gcf()


# Finally some benchmarks
# ----------------

@benchmark $L * $T

#-

@benchmark $Lʰ * $T

