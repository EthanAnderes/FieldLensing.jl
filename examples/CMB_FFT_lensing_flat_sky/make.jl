#src This file generates:        
#src - `example.ipynb`           
#src - `example.md`              
#src                             
#src Build with `julia make.jl`   


using Literate              #src
                            #src
                            #src
config = Dict(                      #src
    "documenter"    => false,       #src
    "keep_comments" => true,        #src
    "execute"       => true,        #src
    "name"          => "example",   #src
    "credit"        => false,       #src
)                                   #src

Literate.notebook(          #src
    "make.jl",              #src
    config=config,          #src
)                           #src
                            #src
Literate.markdown(          #src
    "make.jl",              #src
    config=config,          #src
)                           #src



#=
ENV["JULIA_FFTW_PROVIDER"] = "MKL"
using Pkg
pkg"build FFTW" 
=#

using FFTW
FFTW.set_num_threads(5)

using FieldLensing
using XFields
using FFTransforms
using Spectra
using Interpolations 
using PyPlot # TODO add this to test/Project.toml

clTfun, clϕfun = let
	cld = Spectra.camb_cls(;
        lmax    = 8050, 
        r       = 0.1,
        ωb      = 0.0224567, 
        ωc      = 0.118489, 
        τ       = 0.055, 
        θs      = 0.0103, 
        logA    = 3.043, 
        ns      = 0.968602, 
        Alens   = 1.0, 
        k_pivot = 0.002,
        AccuracyBoost  = 2,
        lSampleBoost   = 4,
        lAccuracyBoost = 4,
    )

	l    = cld[:phi][:ell]
	clT   = cld[:unlen_scalar][:Ctt] ./ cld[:unlen_scalar][:factor_on_cl_cmb] .|> XFields.nan2zero
	clϕ   = cld[:phi][:Cϕϕ] ./ cld[:phi][:factor_on_cl_phi] .|> XFields.nan2zero

    extrap = (l, Cl) -> CubicSplineInterpolation(l, Cl, extrapolation_bc = Line())
    l_extrap = function (l, Cl)
       	iCl = extrap(l, log.(l.^4 .* Cl))
       	return l -> exp(iCl(l)) / l^4
    end

    clTfun = l_extrap(l[3]:l[end], clT[3:end])
    clϕfun = l_extrap(l[3]:l[end], clϕ[3:end])
    
    clTfun, clϕfun
end


trn = let θpix′ = 2.5, nside = 256
    θpix = deg2rad(θpix′/60)
    period = nside * θpix
    𝕨 = r𝕎32(nside, period) ⊗ 𝕎(nside, period)
    ordinary_scale(𝕨)*𝕨
end

CT, Cϕ = let trn = trn
	wvn     = wavenum(trn)
	CTx = clTfun.(wvn)
	Cϕx = clϕfun.(wvn)
	CTx[1] = 0
	Cϕx[1] = 0

	CT = DiagOp(Xfourier(trn, CTx))
	Cϕ = DiagOp(Xfourier(trn, Cϕx))

	CT, Cϕ
end


T, ϕ = let trn = trn, CT = CT, Cϕ = Cϕ
	zTx = randn(eltype_in(trn),size_in(trn)) ./ √Ωx(trn)
	zϕx = randn(eltype_in(trn),size_in(trn)) ./ √Ωx(trn)

	T = √CT * Xmap(trn, zTx)
	ϕ = √Cϕ * Xmap(trn, zϕx)

	T,ϕ
end

#=
fig,ax = subplots(1,2,figsize=(10,4))

y, x = pix(trn) # y indexes rows, x indexes cols 
pcm1 = ax[1].pcolormesh(x,y,T[:])
pcm2 = ax[2].pcolormesh(x,y,ϕ[:])
fig.colorbar(pcm1, ax = ax[1])
fig.colorbar(pcm2, ax = ax[2])
ax[1].set_title(L"T(x)")
ax[2].set_title(L"\phi(x)")

fig.tight_layout()
savefig(joinpath(@__DIR__,"plot1.png")) #src
close() #src
#md # ![plot1](plot1.png)
#nb gcf() 

=#

∇ = let trn = trn 
    k = fullfreq(trn)
    ik1 = Xfourier(trn, im .* k[1]) |> DiagOp
    ik2 = Xfourier(trn, im .* k[2]) |> DiagOp
    (ik1, ik2)
end

L, Lʰ = let trn=trn, nsteps=16, ϕ=ϕ
    vϕ = (∇[1] * ϕ, ∇[2] * ϕ)
    t₀ = 0
    t₁ = 1
    nsteps = 16 
    L  = FieldLensing.Xlense(trn, vϕ, t₀, t₁, nsteps)
    L, L' 
end

lenT1 = L * T
lenT2 = Xmap(trn, FieldLensing.flowRK38(L,T[:]))
T1 = L \ lenT1
T2 = Xmap(trn, FieldLensing.flowRK38(inv(L),lenT2[:]))

#=
T[:] |> matshow
lenT1[:] |> matshow
lenT2[:] |> matshow
(T - lenT1)[:] |> matshow; colorbar()
(T - lenT2)[:] |> matshow; colorbar()
(T - T1)[:] |> matshow; colorbar()
(T - T2)[:] |> matshow; colorbar()
=#


lenʰT1 = Lʰ * T
lenʰT2 = Xmap(trn, FieldLensing.flowRK38(Lʰ,T[:]))
ʰT1 = Lʰ \ lenʰT1
ʰT2 = Xmap(trn, FieldLensing.flowRK38(inv(Lʰ),lenʰT2[:]))
#=
T[:] |> matshow
lenʰT1[:] |> matshow
lenʰT2[:] |> matshow
(T - lenʰT1)[:] |> matshow; colorbar()
(T - lenʰT2)[:] |> matshow; colorbar()
(T - ʰT1)[:] |> matshow; colorbar()
(T - ʰT2)[:] |> matshow; colorbar()
=#



#=
using BenchmarkTools

@benchmark $L * $T
# 25.242 ms, nside 256, Float32, 16 Rk4 steps, 5 threads, MKL

@benchmark $Lʰ * $T
# 26.930 ms, nside 256, Float32, 16 Rk4 steps, 5 threads, MKL

=#