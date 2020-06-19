
######################################
#
# Xlense
#
######################################
# High level holder of information sufficient 
# for generating a lense plan

struct Xlense{Trn<:Transform,Tf,Ti,d}  <: AbstractFlow{Trn,Tf,Ti,d}
	trn::Trn
	v::NTuple{d,Xmap{Trn,Tf,Ti,d}}
	t₀::Int 
	t₁::Int
	nsteps::Int
end

function Xlense(trn::Trn, v::NTuple{d,Xmap{Trn,Tf,Ti,d}}, t₀, t₁, nsteps) where {Tf,Ti,d,Trn<:Transform{Tf,d}}
	Xlense{Trn,Tf,Ti,d}(trn, v, t₀, t₁, nsteps)
end

# Required methods for an AbstractFlow

function Base.inv(L::Xlense{Trn,Tf,Ti,d}) where {Trn<:Transform,Tf,Ti,d}
	Xlense{Trn,Tf,Ti,d}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

function flow(L::Xlense{Trn}, f::Array{Tf,d}) where {Tf,d,Trn<:Transform}
	flowRK4(L,f)
end

function flowRK4(L::Xlense{Trn}, f::Array{Tf,d}) where {Tf,d,Trn<:Transform}
	Lp! = XlensePlan(L)
	odesolve_RK4(Lp!, f, L.t₀, L.t₁, L.nsteps)
end

function flowRK38(L::Xlense{Trn}, f::Array{Tf,d}) where {Tf,d,Trn<:Transform}
	Lp! = XlensePlan(L)
	odesolve_RK38(Lp!, f, L.t₀, L.t₁, L.nsteps)
end


######################################
#
# XlensePlan and AbstractXlensePlan
#
######################################
# XlensePlan is a Low level struct with 
# pre-computed variables for fast 
# computation of the Xlense vector field

abstract type AbstractXlensePlan{Trn<:Transform,Tf,Ti,d} end
# This abstract type to allows to circumvent the fall back method 
# of gradient! 

struct XlensePlan{Trn<:Transform,Tf,Ti,d} <: AbstractXlensePlan{Trn,Tf,Ti,d}
	trn::Trn
	k::  NTuple{d,Array{Tf,d}}    
	vx:: NTuple{d,Array{Tf,d}}  
	∂vx::Matrix{Array{Tf,d}}    
	# --- all the following are storage
	mx:: Matrix{Array{Tf,d}}    
	px:: NTuple{d,Array{Tf,d}}    
	∇y:: NTuple{d,Array{Tf,d}}    
	sk:: Array{Ti,d}
	yk:: Array{Ti,d} 
	function XlensePlan(L::Xlense{Trn,Tf,Ti,d}) where {Trn<:Transform,Tf,Ti,d}
		szf, szi =  size_in(L.trn), size_out(L.trn)
		k   = fullfreq(L.trn)
		vx  = tuple((L.v[i][:] for i=1:d)...)
		∂vx = Array{Tf,d}[(DiagOp(Xfourier(L.trn,im*k[c]))*L.v[r])[:] for r=1:d, c=1:d]
		# --- all the following are storage
		mx  = deepcopy(∂vx)
		px  = deepcopy(vx)
		∇y  = deepcopy(vx)
		sk  = zeros(Ti,szi)
		yk  = zeros(Ti,szi)
		new{Trn,Tf,Ti,d}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
	end
end

# Gradients with XlensePlan
# The default fallback method here is to use fft for gradients.
function gradient!(∇y::NTuple{d,Array{Tf,d}}, y::Array{Tf,d}, Lp::AbstractXlensePlan{Trn}) where {Tf,d,Trn<:Transform{Tf,d}}
	FFT = plan(Lp.trn)
	mul!(Lp.yk, FFT.unscaled_forward_transform, y)
	for i = 1:d
		@inbounds @. Lp.sk = Lp.yk * Lp.k[i] * im * FFT.scale_forward * FFT.scale_inverse
		mul!(∇y[i], FFT.unscaled_inverse_transform, Lp.sk)
	end
end

# Vector field method (d==1). Note: overwrites v
function (Lp::XlensePlan{Trn})(v::Array{Tf,1}, t::Real, y::Array{Tf,1}) where {Tf,Trn<:Transform{Tf,1}}
	@avx @. Lp.mx[1,1]  = 1 / (1 + t * Lp.∂vx[1,1])
	@avx @. Lp.px[1]    = Lp.mx[1,1] * Lp.vx[1]
	gradient!(Lp.∇y, y, Lp) 
	@avx @. v =  px[1] * Lp.∇y[1] # pxⁱ⋅ ∇ⁱ ⋅ yx
end

# Vector field method (d==2). Note: overwrites v
function (Lp::XlensePlan{Trn})(v::Array{Tf,2}, t::Real, y::Array{Tf,2}) where {Tf,Trn<:Transform{Tf,2}}		
	m11,  m12,  m21,  m22  = Lp.mx[1,1],  Lp.mx[1,2],  Lp.mx[2,1],  Lp.mx[2,2]
	∂v11, ∂v12, ∂v21, ∂v22 = Lp.∂vx[1,1], Lp.∂vx[1,2], Lp.∂vx[2,1], Lp.∂vx[2,2]
	p1, p2, v1, v2         = Lp.px[1], Lp.px[2], Lp.vx[1], Lp.vx[2]
	@avx for i ∈ eachindex(y)
		m11[i]  = 1 + t * ∂v22[i] 
		m12[i]  =   - t * ∂v12[i] 
		m21[i]  =   - t * ∂v21[i] 
		m22[i]  = 1 + t * ∂v11[i] 
		dt  = m11[i] * m22[i] - m12[i] * m21[i]
		m11[i] /= dt
		m12[i] /= dt
		m21[i] /= dt
		m22[i] /= dt
		p1[i]  = m11[i] * v1[i] + m12[i] * v2[i]
		p2[i]  = m21[i] * v1[i] + m22[i] * v2[i]
	end
	gradient!(Lp.∇y, y, Lp) 
	@avx @. v =  p1 * Lp.∇y[1] + p2 * Lp.∇y[2] # pxⁱ⋅ ∇ⁱ ⋅ yx
end
