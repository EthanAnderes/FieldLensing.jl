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

# Note: flow(L,f::NTuple) requires Lp! = plan(L) eat NTuples as arguments.
# if you want broadcasting behavior then you need to impliment it in Lp!,
# not by broadcasting flow to the elements of the tuple. 

# input: Array or tuples of Arrays -> output: Array or tuples of Arrays

function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::Union{A, NTuple{n,A}}) where {Tf, Ti, d, n, A<:Array{Tf,d}, Trn<:Transform{Tf}}
	Lp! = plan(L)
	odesolve_RK4(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

function flowRK38(L::AbstractFlow{Trn,Tf,Ti,d}, f::Union{A, NTuple{n,A}}) where {Tf, Ti, d, n, A<:Array{Tf,d}, Trn<:Transform{Tf}}
	Lp! = plan(L)
	odesolve_RK38(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

# AbstractLinearOp interface methods for AbstractFlow
# ----------------------------

# Over-ride this for custom behavior
# This is just a default for Xfields ... lensing is done on the Xmap
function _lmult(O::AbstractFlow, f::Xfield) 
 	lnfd = flow(O, fielddata(MapField(f)))
 	return Xmap(fieldtransform(f), lnfd)
end

# the rest is automatic

function Base.:*(O::AbstractFlow, f::MF) where {MF <: Field}  
	convert(MF, _lmult(O, f))
end

Base.:\(O::AbstractFlow, f::Field) = inv(O) * f

# Base.:inv(O::AbstractLinearOp) # already defined by lensing operators

# LinearAlgebra.adjoint(O::AbstractLinearOp) # already defined by lensing operators


# ODE solvers used by flow 
# -----------------------------
include("ode_solvers.jl")


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
