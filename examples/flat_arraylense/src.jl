

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

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::B) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}}
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

    return trm, ∇!
end


# ------------------------
Cn, Ct, Cϕ, Cω, ΔCϕΔᴴ, ΔCωΔᴴ, Δ, Cv1, Cv2, Cv1v2 = @sblock let trm
	l   = wavenum(trm)
    
    μKarcminT = 5
    cnl = deg2rad(μKarcminT/60)^2  .+ 0 .* l
	Cn  = DiagOp(Xfourier(trm, cnl)) 

    cTl = Spectra.cTl_besselj_approx.(l)
	Ct  = DiagOp(Xfourier(trm, cTl))

    scale_ϕ = 1.25 
	cϕl     = scale_ϕ .* Spectra.cϕl_approx.(l) 
    Cϕ      = DiagOp(Xfourier(trm, cϕl))
    ΔCϕΔᴴ   = DiagOp(Xfourier(trm, l.^4 .* cϕl)) 

    scale_ω = 0.5
    cωl     = scale_ω .* Spectra.cϕl_approx.(l) 
    Cω      = DiagOp(Xfourier(trm, cωl)) 
    ΔCωΔᴴ   = DiagOp(Xfourier(trm, l.^4 .* cωl))

    Δ       = DiagOp(Xfourier(trm, .- l .^ 2))

    k1, k2 = fullfreq(trm)
    Cv1    = DiagOp(Xfourier(trm, abs2.(k1) .* cϕl .+ abs2.(k2) .* cωl))
    Cv2    = DiagOp(Xfourier(trm, abs2.(k2) .* cϕl .+ abs2.(k1) .* cωl))
    Cv1v2  = DiagOp(Xfourier(trm, k1 .* k2 .* (cϕl .- cωl)))

    Cn, Ct, Cϕ, Cω, ΔCϕΔᴴ, ΔCωΔᴴ, Δ, Cv1, Cv2, Cv1v2
end;


logPr, ninv∇logPv = let Cv1=Cv1, Cv2=Cv2, Cv1v2=Cv1v2, Ct=Ct, trm=trm 
    A  = sqrt(inv(Cv1))
    B  = Cv1v2 / Cv1
    AB = sqrt(inv(Cv2 - B))
    
    logPr = function (flnt,fv1,fv2)
        fw1 = A * fv1    
        fw2 = AB * (B * fv1 + fv2)    
        - (dot(flnt, Ct \ flnt) + dot(fw1, fw1) + dot(fw2, fw2)) / 2
    end

    ninv∇logPv = function (fv1, fv2)
        w1 = Cv1*fv1   + Cv1v2*fv2
        w2 = Cv1v2*fv1 + Cv2*fv2
        w1, w2
    end

    logPr, ninv∇logPv
end


divv = function (v)
    ∇v1 = ∇!(v[1])
    ∇v2 = ∇!(v[2])
    ∇v1[1] .+  ∇v2[2]
end

pdivv = function (v)
    ∇v1 = ∇!(v[1])
    ∇v2 = ∇!(v[2])
    ∇v1[2] .-  ∇v2[1]
end

∇vlogPr = function (v)
    ∇vll1 = ∇!(.- (ΔCϕΔᴴ \ Xmap(trm,divv(v)))[:]) 
    ∇vll2 = ∇!(.- (ΔCωΔᴴ \ Xmap(trm,pdivv(v)))[:]) 
    (∇vll1[1] .+ ∇vll2[2], ∇vll1[2] .- ∇vll2[1])
end

#-

n, t, Len, v, vϕ, vω, ϕ, ω = @sblock let trm, ∇!, Cn, Ct, Cϕ, Cω
    t = √Ct * whitemap(trm)
    n = √Cn * whitemap(trm)
    ϕ = √Cϕ * whitemap(trm)
    ω = √Cω * whitemap(trm)
    ### ω = 0 * ω!!!!! 
    vϕ = ∇!(ϕ[:])    
    vω = ∇!(ω[:]) |> x->(x[2], .-x[1])
    v  = (vϕ[1] + vω[1], vϕ[2] + vω[2])     
    Len = v -> FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
    
    n, t, Len, v, vϕ, vω, ϕ, ω
end;

#-

d = Len(v) * t + n

# set initial vcurr

vcurr = map(zero, v)

# update tcurr, vcurr
for rtnn = 1:2
    global tcurr, vcurr

    tcurr, hcurr = pcg(
        f -> (inv(Ct) + inv(Cn)) \ f, 
        #f -> Ct * Cn / (Ct + Cn) * f, 
        f -> Len(vcurr)' * inv(Cn) * Len(vcurr) * f +  inv(Ct) * f,
        Len(vcurr)' * inv(Cn) * (d + √Cn * whitemap(trm)) + inv(Ct^(1//2)) * whitemap(trm),
        nsteps = 50,
        rel_tol = 1e-20,
    )
    
    
    # Gradient update vcurr
    
    ## set transpose flow operators
    τL01 = FieldLensing.τArrayLense(vcurr, (tcurr[:],), ∇!, 0, 1, 16)
    τL10  = inv(τL01)
    
    ## initial transpose flow down to time 0
    # Don't you need a prior term for this one ...?
    ∇logP1 = ((Cn\(d-Len(vcurr)*tcurr))[:],)
    τv0, τf0  = τL10 * (map(zero,vcurr), ∇logP1)
    
    ## update τf0 with ∇logPr(f)
    ## update τv0 with ∇logPr(v)
    τf0[1] .-= (Ct \ tcurr)[:] 
    τv0[1] .+= ∇vlogPr(vcurr)[1] 
    τv0[2] .+= ∇vlogPr(vcurr)[2] 
    
    ## final transpose flow from time 0 to time 1 
    τv1 = (τL01 * (τv0, τf0))[1]    
    
    solver=:LN_SBPLX # :LN_SBPLX, :LN_COBYLA, :LN_NELDERMEAD, :GN_DIRECT_L, :GN_DIRECT_L_RAND
    maxtime=60 
    upper_lim=2.0
    lower_lim=0.0
    inits=0.001 
    
    invΛ∇vcurr = ( (Cv1*Xmap(trm,τv1[1]))[:], (Cv2*Xmap(trm,τv1[2]))[:] )
    # invΛ∇vcurr = map(x->x[:], ninv∇logPv(Xmap(trm,τv1[1]), Xmap(trm,τv1[2])))
    Len_tcurr = Len(vcurr)*tcurr
    T = eltype_in(trm)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = maxtime
    opt.upper_bounds = T[upper_lim]
    opt.lower_bounds = T[lower_lim]
    opt.initial_step = T[inits]
    opt.max_objective = function (β, grad)
        vβ1 = Xmap(trm, vcurr[1] + β[1] * invΛ∇vcurr[1])
        vβ2 = Xmap(trm, vcurr[2] + β[1] * invΛ∇vcurr[2])
        unlen_t = Len((vβ1[:], vβ2[:])) \ Len_tcurr
        logPr(unlen_t, vβ1, vβ2)
    end
    ll_opt, β_opt, = NLopt.optimize(opt,  T[0.000001])
    @show ll_opt, β_opt
    vcurr = ( 
        vcurr[1] + β_opt[1] * invΛ∇vcurr[1],
        vcurr[2] + β_opt[1] * invΛ∇vcurr[2]
    )

end

#=
v[1]     |> matshow; colorbar()
vcurr[1] |> matshow; colorbar()

v[2]     |> matshow; colorbar()
vcurr[2] |> matshow; colorbar()


(Δ \ Xmap(trm, divv(vcurr)))[:] |> matshow; colorbar()
ϕ[:] |> matshow; colorbar()


(Δ \ Xmap(trm, pdivv(vcurr)))[:] |> matshow; colorbar()
ω[:] |> matshow; colorbar()


divv(vcurr) |> matshow; colorbar()
divv(v) |> matshow; colorbar()

pdivv(vcurr) |> matshow; colorbar()
pdivv(v) |> matshow; colorbar()


=#









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


# Benchmark transpose delta lense 
#= --------------------------
L  = FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
T  = t[:]
LT = L  * T
f  = LT
τL   = FieldLensing.τArrayLense(v, (f,), ∇!, 1, 0, 16)
τL′  = FieldLensing.τArrayLense(v, (f,f), ∇!, 1, 0, 16)

τv  = (0 .* v[1], 0 .* v[2])
τf  = (LT .- T,)
τf′ = (LT .- T, LT .- T)

@code_warntype τL * (τv, τf)
@code_warntype τL′ * (τv, τf′)
@benchmark $τL * $((τv, τf))
##  minimum time:  125.916 ms , 256x256, Float64 (8 threads)
@benchmark $τL′ * $((τv, τf′))
## minimum time: 233.014 ms , 256x256, Float64 (8 threads)

#-
pτL! = FieldLensing.plan(τL) 
# ẏ = cat(f, τf, τv...; dims = 3)
# y = cat(f, τf, τv...; dims = 3)
ẏ = tuple(f, τf, τv...)
y = tuple(f, τf, τv...)
@code_warntype pτL!(ẏ, 1, y)
@benchmark ($pτL!)($(ẏ), 1, $y) # 1.5 ms (0.00% GC), 256x256, Float64

=#




# Benchmark Lensing and adjoint lensing
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


## # Test some different ways to compute (∂(x+tv(x))/∂x) \ v
## # --------------------------


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
