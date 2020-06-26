module FieldLensing

using XFields
import FFTransforms
using LinearAlgebra
using LoopVectorization

# AbstractFlow 
# ===============================

export AbstractFlow, flow 

abstract type AbstractFlow{Trn<:Transform,Tf,Ti,d} end

# plan(L), fallback
# ----------------------------
plan(L::AbstractFlow) = L

# flow(L, f)
# -----------------------------

# Default 
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::Array{Tf,d}) where {Tf, Ti, d, Trn<:Transform{Tf,d}}
	Lp! = plan(L)
	odesolve_RK4(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

function flowRK38(L::AbstractFlow{Trn,Tf,Ti,d}, f::Array{Tf,d}) where {Tf, Ti, d, Trn<:Transform{Tf,d}}
	Lp! = plan(L)
	odesolve_RK38(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

# flow(L,f) where f is a Map Field
function flow(L::AbstractFlow, f::MF)  where {MF<:MapField} 
	tr = fieldtransform(f)
	MF(tr, flow(L, f[:]))
end

# flow(L,f) where f is a Fourier Field
function flow(L::AbstractFlow, f::FourierField) # where {Tf, Ti, d, Trn<:Transform{Tf,d}} 
	FourierField(flow(L,MapField(f)))
end

# ODE solvers used by flow 
include("ode_solvers.jl")

# `*` and `\` call flow(L, f)
# -----------------------------

Base.:*(L::AbstractFlow, f) = flow(L,f)
Base.:\(L::AbstractFlow, f) = flow(inv(L),f)

# Pre-installed AbstractFlow types and methods
# =========================================

# ArrayLense
export ArrayLense, ArrayLenseᴴ

include("array_lense/array_lense.jl")
include("array_lense/adjoint_array_lense.jl")

# Xlense and AdjointXlense
export Xlense, AdjointXlense

include("xlense/xlense.jl")
include("xlense/xlense_gradient_plan.jl")

include("xlense/adjoint_xlense.jl")
include("xlense/adjoint_xlense_gradient_plan.jl")



end
