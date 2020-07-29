
# Using Xlense from FieldLensing with HealpixTransforms
# ==========

# Load modules
# ----------
using XFields
using HealpixTransforms
using FieldLensing
using Spectra

using SparseArrays
using LinearAlgebra
using LBblocks: @sblock
using PyPlot
using BenchmarkTools

# Type alias
# --------------
const F64 = Float64
const C64 = Complex{F64}
const FL  = FieldLensing
const HT  = HealpixTransforms

# Simulate a mock CMB temp field and a lensing potential on the sphere
# ===========================

# Set fourier grid parameters
# --------------------------

trn = let nside = 1024 # 512 # 1024 # 2048 
    ℍ0(nside, iter=1)
end

# Compute the spectral matrices which mimic CMB tempurature and lesing potential
# ---------------------------

Ct, Cϕ = @sblock let trn
    l, m = lm(trn)

    cϕl = Spectra.cϕl_approx.(l) 
    cϕl[l .< 2] .= 0
    
    cTl = Spectra.cTl_approx.(l)
    #cTl = Spectra.cTl_besselj_approx.(l)
    #cTl = Spectra.cTl_matern_cov_approx.(l)
    cTl .*= Spectra.knee.(l; ell=50, alpha=2)
    cTl[l .< 2] .= 0

    Ct  = DiagOp(Xfourier(trn, cTl)) 
    Cϕ  = DiagOp(Xfourier(trn, cϕl)) 

    Ct, Cϕ
end;

# Simulate and compute two vectorfields from phi
# ------------------------

T, ϕ = @sblock let trn, Ct, Cϕ
    zTlm = randn(eltype_out(trn),size_out(trn))
    zϕlm = randn(eltype_out(trn),size_out(trn))

    T = √Ct * Xfourier(trn, zTlm)
    ϕ = √Cϕ * Xfourier(trn, zϕlm)

    T, ϕ
end;

#-

vϕ_on_ê, vϕ_on_eθφ = let trn=trn, ϕ=ϕ
    ∇θf, sinθ⁻¹∇φf = HT.∇(ϕ[!], trn)[1:2]
    sinθ = sin.(pix(trn)[1])
    vϕ_on_ê = (Xmap(trn,∇θf), Xmap(trn, sinθ⁻¹∇φf))
    vϕ_on_eθφ = (Xmap(trn,∇θf), Xmap(trn, sinθ⁻¹∇φf ./ sinθ))
    vϕ_on_ê, vϕ_on_eθφ
end;

#-
HT.mollview(T[:], title="CMB temp")


#-
HT.mollview(ϕ[:], title="lensing potential")




# Testing interp of gradients on sphere for lensing 
# ====================

θfull, φfull  = pix(trn)
smallθ = (θfull .< 0.025)
midθ   = (π/2-0.01) .< θfull .< (π/2+0.01)
v1_eθφ   = vϕ_on_eθφ[1][:]
v2_eθφ   = vϕ_on_eθφ[2][:]
v1_ê   = vϕ_on_ê[1][:]
v2_ê   = vϕ_on_ê[2][:];

# The geodesic distance of displacement should be the similar 
# for all θ
displ_geo = function (v1, v2)
    initn̂ = HT.n̂.(θfull,φfull)
    termn̂ = HT.n̂.(θfull .+ v1, φfull .+ v2)
    .- log.(HT.α_n̂.(initn̂, termn̂))
end

# Test vϕ_on_ê
# ---------------------
fig, ax = subplots(1,2)
ax[1].hist(abs.(v2_ê[smallθ]), histtype="step", density=true)
ax[1].hist(abs.(v2_ê[midθ]),   histtype="step", density=true)
ax[2].hist(abs.(v2_eθφ[smallθ]), histtype="step", density=true)
ax[2].hist(abs.(v2_eθφ[midθ]),   histtype="step", density=true)


# these look to have the same magnitude if using raw HT.∇
# ... which seems to imply that vϕ_on_ê are indeed weights on êθ and êφ
v1_ê |> HT.eqbelt |> matshow; colorbar(); 
#-
v2_ê |> HT.eqbelt |> matshow; colorbar(); 


# Test vϕ_on_eθφ
# ---------------------
fig, ax = subplots(1,2, sharey=true)
displ_geo_ê   = displ_geo(v1_ê, v2_ê)
displ_geo_eθφ = displ_geo(v1_eθφ, v2_eθφ)
ax[1].plot(θfull[smallθ], displ_geo_ê[smallθ], ".", alpha = 0.8)
ax[2].plot(θfull[smallθ], displ_geo_eθφ[smallθ], ".", alpha = 0.8)




#-
v1_eθφ |> HT.eqbelt |> matshow; colorbar(); 
#-
v2_eθφ |> HT.eqbelt |> matshow; colorbar(); 



# Lets check that we can get the same vϕ_on_ê, vϕ_on_eθφ on the 
# equitorial belt with discrete increments
# ====================

Tbelt = HT.eqbelt(T[:])
ϕbelt = HT.eqbelt(ϕ[:]);

#-
v1_eθφ_belt  = HT.eqbelt(vϕ_on_eθφ[1][:])
v2_eθφ_belt  = HT.eqbelt(vϕ_on_eθφ[2][:])
v1_ê_belt  = HT.eqbelt(vϕ_on_ê[1][:])
v2_ê_belt  = HT.eqbelt(vϕ_on_ê[2][:]);

#-
θbelt, φbelt = HT.pix_eqbelt(trn.nside)
sinθbelt = sin.(θbelt)
Δcosθ    = mean(diff(cos.(θbelt))) 
nθ       = length(θbelt)
Δφ       = mean(diff(φbelt[:]))
nφ       = length(φbelt)
onesnφm1 = fill(1,nφ-1);

# ∂θ (action by left mult)
#------------------------
#  for healpix on the equitorial belt, cos(θ) is on an even grid.
onesnθm1 = fill(1,nθ-1)
∂cosθ = (1 / (2Δcosθ)) * spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
∂θ = (.- sinθbelt) .* ∂cosθ # using the chain rule and -sin(θ) = (d/dθ) cosθ
∂θ[1,:] .= 0
∂θ[end,:] .= 0;


# ∂φᵀ (action by right mult)
#------------------------
∂φ      = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
## for the periodic boundary conditions
∂φ[1,end] = -1
∂φ[end,1] =  1
## now as a right operator
∂φᵀ = transpose((1 / (2Δφ)) * ∂φ);
## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ

# Apply these to the phi realization
# ----------------------------

∂θϕ = ∂θ * ϕbelt
∂φϕ = ϕbelt * ∂φᵀ

test_v1_eθφ_belt = ∂θϕ
test_v2_eθφ_belt = ∂φϕ ./ abs2.(sinθbelt)
test_v1_ê_belt   = ∂θϕ
test_v2_ê_belt   = ∂φϕ ./ sinθbelt;

#-
fig, ax = subplots(2,1, figsize=(10,8))
pcm1 = ax[1].imshow(test_v1_eθφ_belt)
pcm2 = ax[2].imshow(v1_eθφ_belt)
ax[1].set_title(L"displacement in $\theta$ finite differencing")
ax[2].set_title(L"displacement in $\theta$ from healpix")
fig.colorbar(pcm1, ax = ax[1])
fig.colorbar(pcm2, ax = ax[2])
fig.tight_layout()


#-
fig, ax = subplots(2,1, figsize=(10,8))
pcm1 = ax[1].imshow(test_v2_eθφ_belt)
pcm2 = ax[2].imshow(v2_eθφ_belt)
ax[1].set_title(L"displacement in $\varphi$ finite differencing")
ax[2].set_title(L"displacement in $\varphi$ from healpix")
fig.colorbar(pcm1, ax = ax[1])
fig.colorbar(pcm2, ax = ax[2])
fig.tight_layout()



#-
fig, ax = subplots(2,1, figsize=(10,8))
pcm1 = ax[1].imshow(test_v2_ê_belt)
pcm2 = ax[2].imshow(v2_ê_belt)
ax[1].set_title(L"coeff on $\hat e_{\varphi}$ finite differencing")
ax[2].set_title(L"coeff on $\hat e_{\varphi}$ from healpix")
fig.colorbar(pcm1, ax = ax[1])
fig.colorbar(pcm2, ax = ax[2])
fig.tight_layout()


# Lensing two ways: full sphere with Xlense; equitorial belt with ArrayLense
# ======================================





# Equitorial belt with ArrayLense
# ======================================
# To use ArrayLense we just need to define ∇!

struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y)
    mul!(∇y[2], y, ∇!.∂φᵀ)
    ∇y
end


function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end

# belt displacement field

### The following leads to some systematics at the edges
## vϕbelt = @sblock let ϕ, ∂θ, ∂φᵀ, sinθbelt
##     ϕbelt = HT.eqbelt(ϕ[:])
##     ∂θϕ = ∂θ * ϕbelt
##     ∂φϕ = ϕbelt * ∂φᵀ
##     v1_eθφ_belt = ∂θϕ
##     v2_eθφ_belt = ∂φϕ ./ abs2.(sinθbelt)
##     (v1_eθφ_belt, v2_eθφ_belt)
## end

vϕbelt = @sblock let vϕ_on_eθφ
    (HT.eqbelt(vϕ_on_eθφ[1][:]), HT.eqbelt(vϕ_on_eθφ[2][:]))
end

# Now construct the lense 
L = @sblock let v=vϕbelt, ∂θ, ∂φᵀ,  ∇! = Nabla!(∂θ, ∂φᵀ), nsteps=16,
    t₀ = 0
    t₁ = 1
    FL.ArrayLense(v, ∇!, t₀, t₁, nsteps)
end;

#-
@time lenTbelt = L * Tbelt

#-
lenTbelt |> matshow; colorbar(); 

#-
(lenTbelt .- Tbelt) |> matshow; colorbar(); 

# ### Inverse Lense 

#-
@time Tbelt′ = L \ lenTbelt

#-
(Tbelt′ .- Tbelt) |> matshow; colorbar(); 

#-
@time lenᴴTbelt = L' * Tbelt
lenᴴTbelt |> matshow; colorbar(); 


# Full sphere with Xlense
# ======================================
# To use Xlense we need to define FL.plan and FL.gradient!


let sinθ = sin.(pix(trn)[1])

    ## for ℍ0 we have Xlense{2,...,1} since it's a 2 dimensional field, stored in 1-d Array
    function FL.plan(L::Xlense{2,ℍ0,Tf,Ti,1}) where {Tf<:F64, Ti<:C64}
        szf, szi =  size_in(L.trn), size_out(L.trn)
        k     = lm(L.trn)
        vx    = tuple(L.v[1][:], L.v[2][:])
        
        ∇v1x = HT.∇(L.v[1][!], L.trn)[1:2]
        ∇v2x = HT.∇(L.v[2][!], L.trn)[1:2]
        ∂vx = Array{Tf,1}[Array{Tf,1}(undef,szf) for r=1:2, c=1:2]
        ∂vx[1,1] .= ∇v1x[1]
        ∂vx[2,1] .= ∇v2x[1]
        ∂vx[1,2] .= sinθ.*∇v1x[2] # extra sinθ to get d/dφ
        ∂vx[2,2] .= sinθ.*∇v2x[2] # extra sinθ to get d/dφ

        mx  = deepcopy(∂vx)
        px  = deepcopy(vx)
        ∇y  = deepcopy(vx)
        sk  = zeros(C64,szi)
        yk  = zeros(C64,szi)
        FL.XlensePlan{2,ℍ0,Tf,Ti,1}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
    end

    function FL.gradient!(∇y::NTuple{2,Array{F64,1}}, y::Array{F64,1}, Lp::FL.XlensePlan{2,ℍ0,F64,C64,1})
        f = Xmap(Lp.trn, y)
        ∇θf, sinθ⁻¹∇φf = HT.∇(f[!], Lp.trn)[1:2]
        ∇y[1] .= ∇θf
        ∇y[2] .= sinθ .* sinθ⁻¹∇φf
    end

end

# ... do the one for 
    




# Construct lense
# ---------------

# Now construct the lense

L = @sblock let trn, vϕ=vϕ_on_eθφ, nsteps=16
    t₀ = 0
    t₁ = 1
    FL.Xlense(trn, vϕ, t₀, t₁, nsteps)
end;

# Forward lensing
# ---------------

@time lenT = L * T;

#- 
lenT[:] |> HT.eqbelt |> matshow; colorbar(); 

#- 
HT.mollview(lenT[:], title="lensed")


# Difference between lensed and un-lensed
(T - lenT)[:] |> HT.eqbelt |> matshow; colorbar(); 

#- 
HT.mollview((T - lenT)[:], title="lensed - unlensed")



# Inverse lensing
# ---------------

@time T′ = L \ lenT;

#- 
HT.mollview(T′[:], title="inverse lense of lensed T")
 

#- 
T′[:] |> HT.eqbelt |> matshow; colorbar(); 

# Difference between lensed and un-lensed
(T - T′)[:] |> HT.eqbelt |> matshow; colorbar(); 

#- 
HT.mollview((T - T′)[:], title="(original T) - (inverse lense of lensed T)")






# Compare Equitorial belt lense to full sky lense 
# ======================================


# Here is the lensing of the eqbelt 
lenTbelt |> matshow; colorbar(); 

# Here is the equitorial belt of the full sky lense 
beltlenT = HT.eqbelt(lenT[:]);
beltlenT |> matshow; colorbar(); 

# Here is the difference between the two
(lenTbelt .- beltlenT) |> matshow; colorbar(); 

