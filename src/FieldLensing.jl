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
# Note: I think one can bipass odesolve_RK4 for flow as follows
# function FieldLensing.flow(L::Xlense{Trn}, f::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}
#	FieldLensing.flowRK38(L)
# end

# flow(L,f) where f is a Map Field
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::MF)  where {Tf, Ti, d, Trn<:Transform{Tf,d}, MF<:MapField{Trn,Tf,Ti,d}} 
	tr = fieldtransform(f)
	MF(tr, flow(L, f[:]))
end

# flow(L,f) where f is a Fourier Field
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::FourierField{Trn,Tf,Ti,d})  where {Tf, Ti, d, Trn<:Transform{Tf,d}} 
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

# Xlense
export Xlense
include("Xlense/xlense.jl")
include("Xlense/gradient_plan.jl")


# AdjointXlense
export AdjointXlense
include("AdjointXlense/adjoint_xlense.jl")
include("AdjointXlense/gradient_plan.jl")



end
