

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

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::B) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end

struct Jacobian!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (𝕁!::Jacobian!{Tθ,Tφ})(y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
	y11, y21, y12, y22 = similar(y[1]), similar(y[1]), similar(y[1]), similar(y[1])
	mul!(y11, ∇!.∂θ, y[1])
	mul!(y21, ∇!.∂θ, y[2])
	mul!(y12, y[1], ∇!.∂φᵀ)
	mul!(y22, y[2], ∇!.∂φᵀ)
	y11, y21, y12, y11
end

# -----------------------------------------------
trm, ∇!, 𝕁! = @sblock let Δθ′ = 3.5, Δφ′ = 3.5, nθ = 256, nφ = 256
	## 𝕨      = r𝕎32(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
	𝕨      = r𝕎(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
	trm    = ordinary_scale(𝕨)*𝕨

	onesnθm1 = fill(1,nθ-1)
	∂θ = spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
	## ∂θ[1,:] .= 0
	## ∂θ[end,:] .= 0
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
    𝕁! = Jacobian!(∂θ, ∂φᵀ)

    return trm, ∇!, 𝕁!
end


# ------------------------
function whitemap(trm::T) where T<:Transform
    zx = randn(eltype_in(trm),size_in(trm))
    Xmap(trm, zx ./ √Ωx(trm))
end


Ct, Cϕ, Cn = @sblock let trm, scale_lense = 1.25, μKarcminT = 10
	l   = wavenum(trm)
    
    cTl = Spectra.cTl_besselj_approx.(l)
	cϕl = Spectra.cϕl_approx.(l) 
    cnl = deg2rad(μKarcminT/60)^2  .+ 0 .* l

	Ct  = DiagOp(Xfourier(trm, cTl)) 
    Cϕ  = scale_lense * DiagOp(Xfourier(trm, cϕl)) 
	Cn  = DiagOp(Xfourier(trm, cnl)) 

    Ct, Cϕ, Cn
end;


t, n, ϕ, v = @sblock let trm, ∇!, Ct, Cϕ, Cn
    t = √Ct * whitemap(trm)
    n = √Cn * whitemap(trm)
    ϕ = √Cϕ * whitemap(trm)
    v = ∇!(ϕ[:])    
    t, n, ϕ, v
end;



#= ------------------------
L  = FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
Lᴴ = L'

T   = t[:]
LT  = L  * T
LᴴT = Lᴴ * T

L⁻¹LT  = L \ LT  
L⁻ᴴLᴴT = Lᴴ \ LᴴT

#-
T    |> matshow; colorbar();
LT   |> matshow; colorbar();
T .- LT   |> matshow; colorbar();

#-
T    |> matshow; colorbar();
LᴴT  |> matshow; colorbar();
T .- LᴴT  |> matshow; colorbar();

#
T .- L⁻¹LT	|> matshow; colorbar();
T .- L⁻ᴴLᴴT	|> matshow; colorbar();

#-
@benchmark $L * $T   # 35.966 ms (0.00% GC), 256x256, 16 steps, Float64
@benchmark $Lᴴ * $T  #

=#

# Test transpose delta lense 
# --------------------------
L  = FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
τL = FieldLensing.τArrayLense(v, ∇!, 1, 0, 16)
T  = t[:]
LT = L  * T


f   = LT
τf  = LT .- T
τv  = (0 .* v[1], 0 .* v[2])


#-
@code_warntype τL(f, τf, τv)
@benchmark $τL($f, $τf, $τv)
##  minimum time:  125.916 ms , 256x256, Float64


#-
pτL! = FieldLensing.plan(τL) 
# ẏ = cat(f, τf, τv...; dims = 3)
# y = cat(f, τf, τv...; dims = 3)
ẏ = tuple(f, τf, τv...)
y = tuple(f, τf, τv...)
@code_warntype pτL!(ẏ, 1, y)
@benchmark ($pτL!)($(ẏ), 1, $y) # 1.5 ms (0.00% GC), 256x256, Float64


f_out, τf_out, τv_out = τL(f, τf, τv)
rtn = FieldLensing.odesolve_RK4_tup(pτL!, tuple(f, τf, τv...), τL.t₀, τL.t₁, τL.nsteps)

τv_out[1] |> matshow; colorbar()
v[1] |> matshow; colorbar()



τv_out[2] |> matshow; colorbar()
v[2] |> matshow; colorbar()


τf_out |> matshow; colorbar()
f     |> matshow; colorbar()

f_out |> matshow; colorbar()
f     |> matshow; colorbar()





## # Test some different ways to compute (∂(x+tv(x))/∂x) \ v
## # --------------------------

t, n, ϕ, v = @sblock let trm, ∇!, Ct, Cϕ, Cn
    t = √Ct * whitemap(trm)
    n = √Cn * whitemap(trm)
    ϕ = √Cϕ * whitemap(trm)
    v = ∇!(ϕ[:])    
    t, n, ϕ, v
end;

Lv = FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
d = Lv * t + n

#-
v0       = (v[1] .* 0, v[2] .* 0)
vcurr    = (v[1] .* 0.0, v[2] .* 0.0)
Lvcurr   = FieldLensing.ArrayLense(vcurr, ∇!, 0, 1, 16)
Lvcurr_t = Lvcurr * t
curr_t   = t

τL10 = FieldLensing.τArrayLense(vcurr, ∇!, 1, 0, 16)
τL01 = FieldLensing.τArrayLense(vcurr, ∇!, 0, 1, 16)

#-
τf, τv = τL10(Lvcurr_t[:], (Cn \ (d - Lvcurr_t))[:], v0)[2:3]
τf    .-= (Ct \ curr_t)[:] 
# τv[1] .-= (Cϕ \ Xmap(trm,vcurr[1]))[:] 
# τv[2] .-= (Cϕ \ Xmap(trm,vcurr[2]))[:]
τv = τL01(curr_t[:], τf, τv)[3]

(Cϕ * Xmap(trm, ∇!(τv[1])[1] + ∇!(τv[2])[2]))[:] |> matshow
(Xmap(trm, ∇!(v[1])[1] +  ∇!(v[2])[2]))[:] |> matshow


(√Cϕ * Xmap(trm,τv[1]))[:] |> matshow
v[1]  |> matshow

(√Cϕ * Xmap(trm,τv[2]))[:] |> matshow
v[2]  |> matshow


(Cϕ * Xmap(trm,τv[1] .+ τv[2]))[:] |> matshow
v[1] .+ v[2] |> matshow




#TODO: set up the following for a very basic test of the transpose delta flow.


        # # -------- ∇mϕ
        # ∇logPp, ∇logPϕ = δᵀ_Flowϕ_local(
        #     lnp,  ϕ,                               # parameterize the lense path (f,ϕ)
        #     𝔸ᵀ(ncl \ (dp - 𝔸(lnp))),  zero(ϕ),     # operated on (δf,δϕ)
        #     1, 0, ode_steps
        # )
        # ∇logPp -= ((Cfs + ccurr.r * unit_Cft) \ p) 
        # ∇logPϕ -= inv(ccurr.Aϕ) * (invΣϕ_unit * ϕ)
        # ∇ϕ = δᵀ_Flowϕ_local(
        #     𝔻r * p,      ϕ,        # parameterize the lense path (f,ϕ)
        #     𝔻r \ ∇logPp, ∇logPϕ,   # operated on (δf,δϕ)
        #     0, 1, ode_steps
        # )[2]
        # ∇mϕ = grad_mult * (𝔾Aϕ \ ∇ϕ)










## # Test some different ways to compute (∂(x+tv(x))/∂x) \ v
## # --------------------------
## 
## function test1(p1, p2, t, j11, j21, j12, j22, v1, v2)
## 	@inbounds for i ∈ eachindex(j11)
## 		y = SMatrix{2,2}(1 + t*j11[i], t*j21[i], t*j12[i], 1 + t*j22[i]) \ SVector(v1[i], v2[i])
## 		# y = factorize([1+t*j11[i]  t*j12[i] ; t*j21[i]  1+t*j22[i]]) \ SVector(v1[i], v2[i])
## 		p1[i]  = y[1]
## 		p2[i]  = y[2]
## 	end
## end
## 
## 
## function test2(p1, p2, t, j11, j21, j12, j22, v1, v2)
## 	@avx for i ∈ eachindex(j11)
## 		m11  = 1 + t * j22[i] 
## 		m12  =   - t * j12[i] 
## 		m21  =   - t * j21[i] 
## 		m22  = 1 + t * j11[i] 
## 		dt  = m11 * m22 - m12 * m21
## 		p1[i]  = (m11 * v1[i] + m12 * v2[i]) / dt
## 		p2[i]  = (m21 * v1[i] + m22 * v2[i]) / dt
## 	end
## end
## 
## 
## function test3(p1, p2, t, j11, j21, j12, j22, v1, v2)
## 	@avx for i ∈ eachindex(j11)
## 		m11  = 1 + t * j22[i] 
## 		m12  =   - t * j12[i] 
## 		m21  =   - t * j21[i] 
## 		m22  = 1 + t * j11[i] 
## 		dt   = m11 * m22 - m12 * m21
## 		m11  /= dt
## 		m12  /= dt
## 		m21  /= dt
## 		m22  /= dt
## 		p1[i]  = m11 * v1[i] + m12 * v2[i]
## 		p2[i]  = m21 * v1[i] + m22 * v2[i]
## 	end
## end
## 
## 
## 
## function test4(p1, p2, t, j11, j21, j12, j22, v1, v2)
## 	@avx for i ∈ eachindex(j11)
## 		m11  = 1 + t * j22[i] 
## 		m12  =   - t * j12[i] 
## 		m21  =   - t * j21[i] 
## 		m22  = 1 + t * j11[i]
## 		dt   = m11 * m22 - m12 * m21 
## 		dt′  = m11 * m21 + m12 * m22 
## 		r    = hypot(m12, m22)
## 		c    = m22 / r
## 		s    = m12 / r
## 		p1[i] = (c*v1[i] - s*v2[i]) / (dt/r)
## 		p2[i] = s*v1[i] + c*v2[i] - (dt′/r) * p1[i]
## 	end
## end
## 
## 
## t₀ = 0.9f0
## j11, j21, j12, j22 = 𝕁!(v)
## v1, v2 = v 
## p1, p2 = similar(v1), similar(v1)
## 
## @benchmark test1($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
## @benchmark test2($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
## @benchmark test3($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
## @benchmark test4($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
## 
## 
## t₀ = 3.25f0
## 
## p1a, p2a = similar(v1), similar(v1)
## p1b, p2b = similar(v1), similar(v1)
## p1c, p2c = similar(v1), similar(v1)
## p1d, p2d = similar(v1), similar(v1)
## 
## test1(p1a, p2a, t₀, j11, j21, j12, j22, v1, v2)
## test2(p1b, p2b, t₀, j11, j21, j12, j22, v1, v2)
## test3(p1c, p2c, t₀, j11, j21, j12, j22, v1, v2)
## test4(p1d, p2d, t₀, j11, j21, j12, j22, v1, v2)
## 
## 
## matshow(p1a); colorbar()
## matshow(p1b); colorbar()
## matshow(p1c); colorbar()
## matshow(p1d); colorbar()
## 
## 
## matshow(p2a); colorbar()
## matshow(p2b); colorbar()
## matshow(p2c); colorbar()
## matshow(p2d); colorbar()
## 
## 
## 
## matshow(abs.(p1a .- p1b)); colorbar()
## matshow(abs.(p1a .- p1c)); colorbar()
## matshow(abs.(p1a .- p1d)); colorbar()
## 
