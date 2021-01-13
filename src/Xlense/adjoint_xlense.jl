# AdjointXlense and AdjointXlensePlan
# ===============================================

struct AdjointXlense{m,Trn<:Transform,Tf,Ti,d}  <: AbstractFlow
	trn::Trn
	v::NTuple{m,Xmap{Trn,Tf,Ti,d}}
	t₀::Int 
	t₁::Int
	nsteps::Int	
end

function AdjointXlense(trn::Trn, v::NTuple{m,Xmap{Trn,Tf,Ti,d}}, t₀, t₁, nsteps) where {m,Tf,Ti,d,Trn<:Transform{Tf,d}}
	AdjointXlense{m,Trn,Tf,Ti,d}(trn, v, t₀, t₁, nsteps)
end

function Base.inv(L::AdjointXlense{m,Trn,Tf,Ti,d}) where {m,Tf,Ti,d,Trn<:Transform{Tf,d}}
	AdjointXlense{m,Trn,Tf,Ti,d}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::Xlense{m,Trn,Tf,Ti,d}) where {m,Tf,Ti,d,Trn<:Transform{Tf,d}}
	AdjointXlense{m,Trn,Tf,Ti,d}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::AdjointXlense{m,Trn,Tf,Ti,d}) where {m,Tf,Ti,d,Trn<:Transform{Tf,d}}
	Xlense{m,Trn,Tf,Ti,d}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

struct AdjointXlensePlan{m,Trn<:Transform,Tf,Ti,d} 
	trn::Trn
	k::  NTuple{m,Array{Tf,d}}    
	vx:: NTuple{m,Array{Tf,d}}  
	∂vx::Matrix{Array{Tf,d}}    
	mx:: Matrix{Array{Tf,d}}    # the following are storage
	px:: NTuple{m,Array{Tf,d}}    
	∇y:: NTuple{m,Array{Tf,d}}    
	sk:: Array{Ti,d}
	yk:: Array{Ti,d}  
end

# Vector field method (m==1). Note: overwrites v
function (Lp::AdjointXlensePlan{1,Trn})(v::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {d,Tf,Trn<:Transform{Tf,d}}
	@avx @. Lp.mx[1,1]  = 1 / (1 + t * Lp.∂vx[1,1])
	@avx @. Lp.px[1]    = Lp.mx[1,1] * Lp.vx[1] * y
	gradient!(Lp.∇y, (Lp.px[1],), Lp) 
	@avx @. v =  Lp.∇y[1] # ∇ⁱ ⋅ pxⁱ ⋅ yx
end

# Vector field method (d==2). Note: overwrites v
function (Lp::AdjointXlensePlan{2,Trn})(v::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {d,Tf,Trn<:Transform{Tf,d}}
	m11,  m12,  m21,  m22  = Lp.mx[1,1],  Lp.mx[1,2],  Lp.mx[2,1],  Lp.mx[2,2]
	∂v11, ∂v12, ∂v21, ∂v22 = Lp.∂vx[1,1], Lp.∂vx[1,2], Lp.∂vx[2,1], Lp.∂vx[2,2]
	p1y, p2y, v1, v2         = Lp.px[1], Lp.px[2], Lp.vx[1], Lp.vx[2]
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
		p1y[i]  = (m11[i] * v1[i] + m12[i] * v2[i]) * y[i] # note extra y[i]
		p2y[i]  = (m21[i] * v1[i] + m22[i] * v2[i]) * y[i] # note extra y[i]
	end
	gradient!(Lp.∇y, (p1y, p2y), Lp) 
	@avx @. v =  Lp.∇y[1] + Lp.∇y[2] # ∇ⁱ ⋅ pxⁱ ⋅ yx
end
