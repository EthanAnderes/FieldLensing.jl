# Xlense and XlensePlan
# ===============================================

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

function Base.inv(L::Xlense{Trn,Tf,Ti,d}) where {Tf,Ti,d,Trn<:Transform{Tf,d}}
	Xlense{Trn,Tf,Ti,d}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

struct XlensePlan{Trn<:Transform,Tf,Ti,d}
	trn::Trn
	k::  NTuple{d,Array{Tf,d}}    
	vx:: NTuple{d,Array{Tf,d}}  
	∂vx::Matrix{Array{Tf,d}}    
	mx:: Matrix{Array{Tf,d}}    # the following are storage
	px:: NTuple{d,Array{Tf,d}}    
	∇y:: NTuple{d,Array{Tf,d}}    
	sk:: Array{Ti,d}
	yk:: Array{Ti,d} 
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

