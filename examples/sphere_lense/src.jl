
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

trn = @sblock let T=Float64
    spin = 0
    𝕊(T, 4*512, 5*512 - 1, spin)
end


#-

@show Δθarcmin = Δθ′(trn)
@show Δφarcmin = Δφ′(trn);

#-

l, m = lm(trn);

#- 

l 

#- 

m


# Compute the spectral matrices which mimic CMB tempurature and lesing potential
# ------------------------------

Ct, Cϕ = @sblock let trn
    l, m   = lm(trn)

    cϕl = Spectra.cϕl_approx.(l) 
    
    cTl = Spectra.cTl_approx.(l)
    ## cTl = Spectra.cTl_besselj_approx.(l)
    ## cTl = Spectra.cTl_matern_cov_approx.(l)
    cTl .*= Spectra.knee.(l; ell=50, alpha=2)
    cTl[l .< 2] .= 0

    cϕl[l .< 2] .= 0

    Ct  = DiagOp(Xfourier(trn, cTl)) 
    Cϕ  = DiagOp(Xfourier(trn, cϕl)) 

    Ct, Cϕ
end;

# Simulate T and ϕ fields
# ---------------

T, ϕ = @sblock let trn, Ct, Cϕ
    zTlm = SphereTransforms.white_fourier(trn)
    zϕlm = SphereTransforms.white_fourier(trn)

    T = √Ct * Xfourier(trn, zTlm)
    ϕ = √Cϕ * Xfourier(trn, zϕlm)

    T, ϕ
end;

#- 
T[:] |> matshow; colorbar();

#-
ϕ[:] |> matshow; colorbar();





# Equitorial belt with ArrayLense
# ======================================
# To use ArrayLense we just need to define ∇!

struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end

# Construct ∂θ (action by left mult)
#------------------------
#  for healpix on the equitorial belt, cos(θ) is on an even grid.

# using SphereTransforms.FastTransforms: chebyshevpoints
# cosθ = chebyshevpoints(Float64, trn.nθ; kind=1)
∂θ = @sblock let trn 
    onesnθm1 = fill(1,trn.nθ-1)
    ∂θ = (1 / (2Δθ(trn))) * spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    ∂θ[1,:] .= 0
    ∂θ[end,:] .= 0
    ∂θ
end


# Construct ∂φᵀ (action by right mult)
#------------------------

∂φᵀ = @sblock let trn 
    onesnφm1 = fill(1,trn.nφ-1)
    ∂φ      = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
    ## for the periodic boundary conditions
    ∂φ[1,end] = -1
    ∂φ[end,1] =  1
    ## now as a right operator
    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
    ∂φᵀ = transpose((1 / (2Δφ(trn))) * ∂φ);
    ∂φᵀ
end

# belt displacement field
### The following leads to some systematics at the edges
vϕbelt = @sblock let trn, ϕ, ∂θ, ∂φᵀ
    θ = pix(trn)[1]
    #sinθ = sin.(θ)
    #cscθ = csc.(θ) # 1/sinθ
    sin⁻²θ = 1 .+ (cot.(θ)).^2 # = cscθ^2

    ϕbelt = ϕ[:]
    ∂θϕ = ∂θ * ϕbelt
    ∂φϕ = ϕbelt * ∂φᵀ
    v1_eθφ_belt = ∂θϕ
    v2_eθφ_belt = ∂φϕ .* sin⁻²θ
    (v1_eθφ_belt, v2_eθφ_belt)
end

# Now construct the lense 
L = @sblock let v=vϕbelt, ∂θ, ∂φᵀ,  ∇! = Nabla!(∂θ, ∂φᵀ), nsteps=16
    t₀ = 0
    t₁ = 1
    FieldLensing.ArrayLense(v, ∇!, t₀, t₁, nsteps)
end;

#-
Tbelt = T[:]
@time lenTbelt = L * Tbelt

#-
lenTbelt[50:end-50,:] |> matshow; colorbar();

#-
(lenTbelt .- Tbelt)[50:end-50,:] |> matshow; colorbar();

# ### Inverse Lense 

#-
@time Tbelt′ = L \ lenTbelt

#-
(Tbelt′ .- Tbelt)[100:end-100,:] |> matshow; colorbar();



#-
@benchmark $L * $T
@benchmark $(L') * $T

(L' * T)[:] |> matshow

# FFT in azimuth with ArrayLense
# ======================================
# To use ArrayLense we just need to define ∇!

struct Nabla!′{Tθ,T1φ,T2φ,T3φ}
    ∂θ::Tθ
    planFFT::T1φ
    ikφ::T2φ
    ak::T3φ
end

function (∇!::Nabla!′{Tθ,T1φ,T2φ,T3φ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,T1φ,T2φ,T3φ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y[1])

    mul!(∇!.ak, ∇!.planFFT.unscaled_forward_transform, y[2])
    @inbounds @. ∇!.ak = ∇!.ak * ∇!.ikφ * ∇!.planFFT.scale_forward * ∇!.planFFT.scale_inverse
    mul!(∇y[2], ∇!.planFFT.unscaled_inverse_transform, ∇!.ak)
    ∇y
end

function (∇!::Nabla!′{Tθ,T1φ,T2φ,T3φ})(∇y::NTuple{2,A}, y::A) where {Tθ,T1φ,T2φ,T3φ,Tf,A<:Array{Tf,2}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!′{Tθ,T1φ,T2φ,T3φ})(y::A) where {Tθ,T1φ,T2φ,T3φ,Tf,A<:Array{Tf,2}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end

#-
𝕨     = 𝕀(trn.nθ) ⊗ r𝕎(trn.nφ, 2π)
plan𝕨 = FFTransforms.plan(𝕨)
kφ    = FFTransforms.freq(𝕨)[2]' |> Array
ak    = zeros(eltype_out(𝕨), size_out(𝕨))
∇!′   = Nabla!′(∂θ, plan𝕨, im .* kφ, ak)

#-
vϕbelt′ = @sblock let ∇!′, trn, ϕ
    θ = pix(trn)[1]
    sin⁻²θ = 1 .+ (cot.(θ)).^2 # = cscθ^2

    ϕbelt = ϕ[:]
    vϕ′ = ∇!′(ϕbelt)
    (vϕ′[1], vϕ′[2] .* sin⁻²θ)
end

# Now construct the lense 
L′ = @sblock let v=vϕbelt′, ∇!′, nsteps=16
    t₀ = 0
    t₁ = 1
    FieldLensing.ArrayLense(v, ∇!′, t₀, t₁, nsteps)
end;

#-
Tbelt = T[:]
@time lenTbelt′ = L′ * Tbelt

#-
lenTbelt′[250:end-250,:] |> matshow; colorbar();

#-
(Tbelt - lenTbelt′)[250:end-250,:] |> matshow; colorbar();

#-
(lenTbelt - lenTbelt′)[250:end-250,:] |> matshow; colorbar();

#- 
@time Tbelt′′ = L′ \ lenTbelt′

# See how well forward, then backward lensing (with fft in azimuth) does 
# for recovering the original field
(Tbelt′′ - Tbelt)[500:end-500,:] |> matshow; colorbar();


# Compare with how well the map space operator does
(Tbelt′ - Tbelt)[500:end-500,:] |> matshow; colorbar();


