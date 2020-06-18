module FieldLensing

using XFields
using FFTransforms
using LinearAlgebra
using LoopVectorization


#####################################
#
# AbstractFlow 
#
#####################################

export AbstractFlow, flow 

abstract type AbstractFlow{Trn<:Transform,Tf,Ti,d} end

# fallback flow(L,Array)
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::Array{Tf,d}) where {Tf,Ti,d,Trn<:Transform{Tf,d}} 
	error("no method `flow(L,f)` method found")
end 

# flow(L,f) where f Map Field
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::MF)  where {Tf,Ti,d, Trn<:Transform{Tf,d}, MF<:MapField{Trn,Tf,Ti,d}} 
	tr = fieldtransform(f)
    MF(tr, flow(L, f[:]))
end

# L * f where f Fourier Field
function flow(L::AbstractFlow{Trn,Tf,Ti,d}, f::FourierField{Trn,Tf,Ti,d})  where {Tf,Ti,d, Trn<:Transform{Tf,d}} 
	FourierFlow(flow(L,MapField(f)))
end

# L * f and L \ f for all cases above
Base.:*(L::AbstractFlow, f) = flow(L,f)
Base.:\(L::AbstractFlow, f) = flow(inv(L),f)

#####################################
#
# Subtypes of AbstractFlow and methods
#
#####################################

# ODE solvers 
include("ode_solvers.jl")

# Xlense type
export Xlense
include("Xlense/xlense.jl")


end
