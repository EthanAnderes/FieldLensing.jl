
# Using Xlense from FieldLensing with SphereTransforms for XFields
# ==========

import FFTW
FFTW.set_num_threads(5)

using FieldLensing
using FieldLensing: ArrayLense

using SphereTransforms
using XFields
using Spectra

import FFTransforms
using FFTransforms: r𝕎, 𝕎, 𝕀, ⊗, ordinary_scale

using SparseArrays
using LinearAlgebra
using LBblocks: @sblock
using PyPlot
using BenchmarkTools


# Set the Xfield transform 
# ----------

trm = @sblock let T=Float64
    spin = 0
    𝕊(T, 4*512, 5*512 - 1, spin)
end


#-

@show Δθarcmin = Δθ′(trm)
@show Δφarcmin = Δφ′(trm);

#-

l, m = lm(trm);

#- 

l 

#- 

m


# Compute the spectral matrices which mimic CMB tempurature and lesing potential
# ------------------------------

Ct, Cϕ = @sblock let trm
    l, m   = lm(trm)

    cϕl = Spectra.cϕl_approx.(l) 
    
    cTl = Spectra.cTl_approx.(l)
    ## cTl = Spectra.cTl_besselj_approx.(l)
    ## cTl = Spectra.cTl_matern_cov_approx.(l)
    cTl .*= Spectra.knee.(l; ell=50, alpha=2)
    cTl[l .< 2] .= 0

    cϕl[l .< 2] .= 0

    Ct  = DiagOp(Xfourier(trm, cTl)) 
    Cϕ  = DiagOp(Xfourier(trm, cϕl)) 

    Ct, Cϕ
end;

# Simulate T and ϕ fields
# ---------------

T, ϕ = @sblock let trm, Ct, Cϕ
    zTlm = SphereTransforms.white_fourier(trm)
    zϕlm = SphereTransforms.white_fourier(trm)

    T = √Ct * Xfourier(trm, zTlm)
    ϕ = √Cϕ * Xfourier(trm, zϕlm)

    T, ϕ
end;

#- 
T[:] |> matshow; colorbar();

#-
ϕ[:] |> matshow; colorbar();





# Equitorial belt with ArrayLense
# ======================================
# To use ArrayLense we just need to define ∇!

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


# Construct ∂θ (action by left mult)
#------------------------
#  for healpix on the equitorial belt, cos(θ) is on an even grid.

# using SphereTransforms.FastTransforms: chebyshevpoints
# cosθ = chebyshevpoints(Float64, trm.nθ; kind=1)
∂θ = @sblock let trm 
    Δθℝ = Δθ(trm)
    ∂θ′ = spdiagm(
            -2 => fill( 1,trm.nθ-2),
            -1 => fill(-8,trm.nθ-1),
             1 => fill( 8,trm.nθ-1),
             2 => fill(-1,trm.nθ-2),
            )
    ∂θ′[1,end]   =  -8
    ∂θ′[1,end-1] =  1
    ∂θ′[2,end]   =  1

    ∂θ′[end,1]   =  8
    ∂θ′[end,2]   = -1
    ∂θ′[end-1,1] = -1

    ∂θ = (1 / (12Δθℝ)) * ∂θ′
    ## return (∂θ - ∂θ') / 2 
    return ∂θ 
end


# Construct ∂φᵀ (action by right mult)
#------------------------

∂φᵀ = @sblock let trm 
    Δφℝ = Δφ(trm)
    ∂φ  = spdiagm(
            -2 => fill( 1,trm.nφ-2),
            -1 => fill(-8,trm.nφ-1),
             1 => fill( 8,trm.nφ-1),
             2 => fill(-1,trm.nφ-2),
            )
    ∂φ[1,end]   =  -8
    ∂φ[1,end-1] =  1
    ∂φ[2,end]   =  1

    ∂φ[end,1]   =  8
    ∂φ[end,2]   =  -1
    ∂φ[end-1,1] =  -1

    ∂φᵀ = transpose((1 / (12Δφℝ)) * ∂φ)
    return ∂φᵀ 
end

# belt displacement field
### The following leads to some systematics at the edges
vϕbelt = @sblock let trm, ϕ, ∂θ, ∂φᵀ
    θ = pix(trm)[1]
    #sinθ = sin.(θ)
    #cscθ = csc.(θ) # 1/sinθ
    sin⁻²θ = csc.(θ).^2

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


struct Pix1dFFTNabla!{Tθ,TW,Tik,Tx} <: FieldLensing.Gradient{2}
    ∂θ::Tθ
    planW::TW
    ikφ::Tik
    sk::Tik
    sx::Tx
end

function LinearAlgebra.adjoint(∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx}) where {Tθ,TW,Tik,Tx}
    return Pix1dFFTNabla!{Tθ,TW,Tik,Tx}(
        ∇!.∂θ',
        ∇!.planW, 
        .- ∇!.ikφ,
        similar(∇!.sk),
        similar(∇!.sx),
    )
end

function Pix1dFFTNabla!(∂θ, w::𝕎{Tf}) where Tf
    wφ = 𝕀(w.sz[1]) ⊗ 𝕎(Tf, w.sz[2:2], w.period[2:2])
    planW = FFTransforms.plan(wφ)
    c_forFFTNabla = Tf(planW.scale_forward * planW.scale_inverse)

    ∇! = Pix1dFFTNabla!(
        ∂θ,
        planW, 
        im .* FFTransforms.fullfreq(wφ)[2] .* c_forFFTNabla,
        Array{eltype_out(wφ)}(undef,size_out(wφ)),
        Array{eltype_in(wφ)}(undef,size_in(wφ)),
    )

    return ∇!
end 

function (∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx})(des, y, ::Val{1}) where {Tθ,TW,Tik,Tx}
    mul!(des, ∇!.∂θ, y)
end

function (∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx})(des, y, ::Val{2}) where {Tθ,TW,Tik,Tx}
    @inbounds ∇!.sx .= y
    mul!(∇!.sk, ∇!.planW.unscaled_forward_transform, ∇!.sx)
    @inbounds ∇!.sk .*= ∇!.ikφ
    mul!(des, ∇!.planW.unscaled_inverse_transform, ∇!.sk)
end



#-
𝕨     = 𝕀(trm.nθ) ⊗ r𝕎(trm.nφ, 2π)
∇!′   = Pix1dFFTNabla!(∂θ, 𝕨)

#-
vϕbelt′ = @sblock let ∇!′, trm, ϕ
    θ = pix(trm)[1]
    sin⁻²θ = csc.(θ).^2

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


