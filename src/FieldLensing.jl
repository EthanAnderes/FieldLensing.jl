module FieldLensing

using XFields
using XFields: AbstractLinearOp
import FFTransforms
using LinearAlgebra
using LoopVectorization

# AbstractFlow 
# ===============================

export AbstractFlow, flow 

abstract type AbstractFlow <: AbstractLinearOp end

# Requirements for L <: AbstractFlow
# ======================================================================

# * needs L.t₀, L.t₁, L.nsteps

# * Over writing methods for either L(ẏ,t,y) or plan(L)(ẏ,t,y)
# where y and ẏ are arrays or tuples of arrays.

# Base.:inv(L::AbstractFlow) 

# LinearAlgebra.adjoint(O::AbstractFlow) 


# Optional interface methods for constructing an AbstractFlow 
# ======================================================================

# extract data from the field so it can be passed to flow
# flow_field(L, f) -> ff
flow_field(L::AbstractFlow, f::Field) = MapField(f)

# flow_data then processes ff by extracting the data to be passed to flow 
# flow_data(L, ff) -> ffd 
flow_data(L::AbstractFlow, ff::Field) = (fielddata(ff),)

# fallback. Use this to convert the minimal information in L to something that operates
plan(L::AbstractFlow) = L
# each specific abstract flow now needs to impliment how a planned flow generates a vector field 

# low level action of the flow on Arrays or tuples of arrays.
# returns another Array or tuple of arrays.
# flow(L, ffd) -> ln_ffd
function flow(L::AbstractFlow, ffd::Union{A, NTuple{n,A}}) where {n, A<:AbstractArray}
	Lp! = plan(L)
	odesolve_RK4(Lp!, ffd, L.t₀, L.t₁, L.nsteps)
end
# Note: flow(L,f::NTuple) requires Lp! = plan(L) eat NTuples as arguments.
# if you want broadcasting behavior then you need to impliment it in Lp!,
# not by broadcasting flow to the elements of the tuple. 

# this is applied to the output of flow and used to reconstruct the field type
# the original input f is also passed 
# flow_reconstruct(L, ff, ln_ffd
function flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::AbstractArray) where {MF<:Field}
	MF(fieldtransform(ff), ln_ffd)
end

function flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::NTuple{1,A}) where {MF<:Field, A<:AbstractArray}
	MF(fieldtransform(ff), ln_ffd[1])
end

function flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::NTuple{n,A}) where {MF<:Field, n, d, T, A<:AbstractArray{T, d}}
	MF(fieldtransform(ff), cat(ln_ffd...;dims = d+1))
end


# Here are interface methods for AbstractLinearOp
# ======================================================================

# lensing of fields that does not convert back to input basis type
# this is used when chaining AbstractLinearOp's to avoid un-necessary intermideiate conversion
@inline function _lmult(L::AbstractFlow, f::Field)
	ff     = flow_field(L, f) # convert field to the appropriate basis
 	ffd    = flow_data(L, ff) # extract data from converted field 
 	ln_ffd = flow(L, ffd) # flow the array or tuple of arrays
 	flow_reconstruct(L, ff, ln_ffd)
end

# lensing which additionall
function Base.:*(L::AbstractFlow, f::MF) where {MF <: Field}  
	convert(MF, _lmult(L, f))
end

Base.:\(L::AbstractFlow, f::Field) = inv(L) * f

Base.:inv(L::AbstractFlow) = error("you forgot to define inv for your AbstractFlow")

LinearAlgebra.adjoint(O::AbstractFlow) = error("you forgot to define adjoint for your AbstractFlow")


# Extras 
# ----------------------------
Base.getindex(f::Field, L::AbstractFlow) = flow_data(L, flow_field(L, f))


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
