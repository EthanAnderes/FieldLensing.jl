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

using FieldLensing
using XFields
using HealpixTransforms
using CMBspectra

using Interpolations 
using PyPlot 

using LBblocks



# Hook into FieldLensing with Harmonic transforms
# ---------------------------------------

# Darn ... this is going to be a problem since ∂vx
# Perhaps it is best to make ℍ0{Tf} be a dimension 2 operator ...


function plan(L::Xlense{Trn,Tf,Ti,1}) where {Tf,Ti,d,Trn<:ℍ0{Tf}}
    szf, szi =  size_in(L.trn), size_out(L.trn)
    k   = lm(L.trn)
    vx  = tuple((L.v[i][:] for i=1:d)...)
    
    ∂vx = Array{Tf,d}[(DiagOp(Xfourier(L.trn,im*k[c]))*L.v[r])[:] for r=1:d, c=1:d]
    
    mx  = deepcopy(∂vx)
    px  = deepcopy(vx)
    ∇y  = deepcopy(vx)
    sk  = zeros(Ti,szi)
    yk  = zeros(Ti,szi)
    XlensePlan{Trn,Tf,Ti,d}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
end

function gradient!(∇y::NTuple{d,Array{Tf,1}}, y::Array{Tf,1}, Lp::XlensePlan{Trn}) where {Tf,d,Trn<:ℍ0{Tf}}
    FFT = FFTransforms.plan(Lp.trn)
    mul!(Lp.yk, FFT.unscaled_forward_transform, y)
    for i = 1:d
        @inbounds @. Lp.sk = Lp.yk * Lp.k[i] * im * FFT.scale_forward * FFT.scale_inverse
        mul!(∇y[i], FFT.unscaled_inverse_transform, Lp.sk)
    end
end


# ...
# ---------------------------------------

clTfun, clϕfun = let
	cld = CMBspectra.camb_cls(;
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


trn = @sblock let nside = 1024 # 2048
    ℍ0(nside, iter=0)
end

CT, Cϕ = @sblock let trn, clTfun, clϕfun
    l, m = lm(trn)
    CTlm = clTfun.(l)
    Cϕlm = clϕfun.(l)
    CTlm[l .<= 2] .= 0
    Cϕlm[l .<= 2] .= 0

    CT = DiagOp(Xfourier(trn, CTlm))
    Cϕ = DiagOp(Xfourier(trn, Cϕlm))

    CT, Cϕ
end


T, ϕ = @sblock let trn, CT, Cϕ
    zTlm = randn(eltype_out(trn),size_out(trn))
    zϕlm = randn(eltype_out(trn),size_out(trn))
    T  = √CT * Xfourier(trn, zTlm)
    ϕ  = √Cϕ * Xfourier(trn, zϕlm)

	Xmap(T), Xmap(ϕ)
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

∇ = @sblock let trn
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
    eqT,  = HealpixTransforms.get_eq_belt(T[:])
    eqE,  = HealpixTransforms.get_eq_belt(E[:])
    eqB,  = HealpixTransforms.get_eq_belt(B[:])
    eqϕ,  = HealpixTransforms.get_eq_belt(ϕ[:])


    eqT |> matshow; colorbar()
    eqE |> matshow; colorbar()
    eqB |> matshow; colorbar()
    eqQ |> matshow; colorbar()
    eqU |> matshow; colorbar()
    eqϕ |> matshow; colorbar()


=#



#=
using BenchmarkTools

@benchmark $L * $T
# 25.242 ms, nside 256, Float32, 16 Rk4 steps, 5 threads, MKL

@benchmark $Lʰ * $T
# 26.930 ms, nside 256, Float32, 16 Rk4 steps, 5 threads, MKL

=#