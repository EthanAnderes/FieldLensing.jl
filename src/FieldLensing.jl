module FieldLensing

using XFields
using XFields: AbstractLinearOp
import FFTransforms
using LinearAlgebra
using LoopVectorization

# AbstractFlow 
# ===============================

export AbstractFlow, flow 

abstract type AbstractFlow{Trn<:Transform,Tf,Ti,d} <: AbstractLinearOp end

# plan(L), fallback
# ----------------------------
plan(L::AbstractFlow) = L

# flow(L, f) where f isa Array or a tuple of arrays
# -----------------------------

# TODO: you need to update array_lense and adjoint_array_lense to work on tuples

# Behavior on Arrays or tuples of Arrays
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::Union{A, NTuple{n,A}}) where {Tf, Ti, d, n, A<:Array{Tf,d}, Trn<:Transform{Tf,d}}
	Lp! = plan(L)
	odesolve_RK4(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

function flowRK38(L::AbstractFlow{Trn,Tf,Ti,d}, f::Union{A, NTuple{n,A}})where {Tf, Ti, d, n, A<:Array{Tf,d}, Trn<:Transform{Tf,d}}
	Lp! = plan(L)
	odesolve_RK38(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

# flow(L, f) where f isa field or a tuple of fields
# -----------------------------

# flow(L,f) where f is a Map Field
function flow(L::AbstractFlow, f::MF) where {MF<:MapField} 
	tr = fieldtransform(f)
	MF(tr, flow(L, f[:]))
end

# flow(L,f) where f is a Fourier Field
function flow(L::AbstractFlow, f::FourierField) # where {Tf, Ti, d, Trn<:Transform{Tf,d}} 
	FourierField(flow(L,MapField(f)))
end

# Now over general tuples
function flow(L::AbstractFlow, f::NTuple{n,Field}) where {n}
	fmapf   = map(MapField, f)
	fmapx   = map(fielddata, fmapf)
	lnfmapx = flow(L, fmapx)
	return map(lnfmapx, f) do lnfx, fi 
		convert(typeof(fi), MF(fieldtransform(fi), lnfx))
	end
end

# ODE solvers used by flow 
# -----------------------------
include("ode_solvers.jl")

# `*` and `\` call flow(L, f)
# -----------------------------

Base.:*(L::AbstractFlow, f::Union{Field,Array}) = flow(L,f)
Base.:\(L::AbstractFlow, f::Union{Field,Array}) = flow(inv(L),f)

# Pre-installed AbstractFlow types and methods
# =========================================

# ArrayLense

include("array_lense/gradient.jl")
include("array_lense/array_lense.jl")
include("array_lense/adjoint_array_lense.jl")
include("array_lense/transpose_delta_array_lense.jl")

# Xlense and AdjointXlense

include("xlense/xlense.jl")
include("xlense/xlense_gradient_plan.jl")

include("xlense/adjoint_xlense.jl")
include("xlense/adjoint_xlense_gradient_plan.jl")



end
