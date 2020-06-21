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
F64 = Float64
C64 = Complex{F64}
# for ℍ0 we have Xlense{2,...,1} since it's a 2 dimensional field, stored in 1-d Array
function FieldLensing.plan(L::Xlense{2,ℍ0,F64,C64,1})
    szf, szi =  size_in(L.trn), size_out(L.trn)
    k     = lm(L.trn)
    vx    = tuple(L.v[1][:], L.v[1][:])
    
    ∇v1x = HealpixTransforms.∇(L.v[1][!], L.trn)[1:2]
    ∇v2x = HealpixTransforms.∇(L.v[2][!], L.trn)[1:2]
    sinθ = sin.(pix(L.trn)[1])
    ∂vx = Array{F64,1}[Array{F64,1}(undef,szf) for r=1:2, c=1:2]
    ∂vx[1,1] .= ∇v1x[1]
    ∂vx[2,1] .= ∇v2x[1]
    ∂vx[1,2] .= sinθ.*∇v1x[2] 
    ∂vx[2,2] .= sinθ.*∇v2x[2] 

    mx  = deepcopy(∂vx)
    px  = deepcopy(vx)
    ∇y  = deepcopy(vx)
    sk  = zeros(C64,szi)
    yk  = zeros(C64,szi)
    FieldLensing.XlensePlan{2,ℍ0,F64,C64,1}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
end

function FieldLensing.gradient!(∇y::NTuple{2,Array{F64,1}}, y::Array{F64,1}, Lp::FieldLensing.XlensePlan{2,ℍ0,F64,C64,1})
    f = Xmap(Lp.trn, y)
    ∇θf, sinθ⁻¹∇φf = HealpixTransforms.∇(f[!], Lp.trn)[1:2]
    ∇y[1] .= ∇θf
    sinθ = sin.(pix(L.trn)[1])
    ∇y[2] .= sinθ .* sinθ⁻¹∇φf
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

	T, ϕ
end

#=

eqT,  = HealpixTransforms.get_eq_belt(T[:])
eqϕ,  = HealpixTransforms.get_eq_belt(ϕ[:])
eqT |> matshow; colorbar()
eqϕ |> matshow; colorbar()

savefig(joinpath(@__DIR__,"plot1.png")) #src
close() #src
#md # ![plot1](plot1.png)
#nb gcf() 

=#

vϕ = @sblock let trn, ϕ
    ∇θf, sinθ⁻¹∇φf = HealpixTransforms.∇(ϕ[!], trn)[1:2]
    sinθ = sin.(pix(trn)[1])
    Xmap(trn,∇θf), Xmap(trn,sinθ⁻¹∇φf ./ sinθ)
end

#=
eqvϕ1,  = HealpixTransforms.get_eq_belt(vϕ[1][:])
eqvϕ2,  = HealpixTransforms.get_eq_belt(vϕ[2][:])
eqvϕ1 |> matshow; colorbar()
eqvϕ2 |> matshow; colorbar()
=#


L, Lʰ = @sblock let trn, vϕ, nsteps=16
    t₀ = 0
    t₁ = 1
    L  = FieldLensing.Xlense(trn, vϕ, t₀, t₁, nsteps)
    L, L' 
end

@time lenT1 = L * T

# lenT2 = Xmap(trn, FieldLensing.flowRK38(L,T[:]))
# T1 = L \ lenT1
# T2 = Xmap(trn, FieldLensing.flowRK38(inv(L),lenT2[:]))
#=

@sblock let T′ = lenT1, T
    eqT,  = HealpixTransforms.get_eq_belt(T[:])
    eqT′,  = HealpixTransforms.get_eq_belt(T′[:])
    eqT[1:500,1:500]   |> matshow; colorbar()
    eqT′[1:500, 1:500] |> matshow; colorbar()
    eqT .- eqT′  |> matshow; colorbar()
end 

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