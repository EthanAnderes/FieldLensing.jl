
# Adjoint array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Need to define a struct with an instance 
# that can operate on arguments as follows 
#  ∇!(∇y::NTuple{m,A}, y::NTuple{m,A}) where {Tf,d, A<:Array{Tf,d}}

# ArrayLenseᴴ
# --------------------------------
struct ArrayLenseᴴ{m,Tf,d,Tg,Tt<:Real}  <: AbstractFlow{XFields.Id{Tf,d},Tf,Tf,d}
	v::NTuple{m,Array{Tf,d}}
	∇!::Tg  
	t₀::Tt
	t₁::Tt
	nsteps::Int
	function ArrayLenseᴴ(v::NTuple{m,Array{Tf,d}}, ∇!::Tg, t₀::Tt, t₁::Tt, nsteps::Int) where {m,Tf,d,Tg,Tt<:Real}
		new{m,Tf,d,Tg,Tt}(v, ∇!, t₀, t₁, nsteps)
	end
end

function Base.inv(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLenseᴴ(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::ArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLenseᴴ{m,Tf,d,Tg,Tt}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLense{m,Tf,d,Tg,Tt}(L.trn, L.v, L.t₁, L.t₀, L.nsteps)
end


# ArrayLenseᴴPlan
# --------------------------------
struct ArrayLenseᴴPlan{m,Tf,d,Tg,Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}    
end

function plan(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm  = deepcopy(∂v)
	p   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	ArrayLenseᴴPlan{m,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, ∇y)
end

# Vector field method (m==1). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{1,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}
	@avx @. Lp.mm[1,1]  = 1 / (1 + t * Lp.∂v[1,1])
	# the following is p⋅y now
	@avx @. Lp.p[1]    = Lp.mm[1,1] * Lp.v[1] * y
	Lp.∇!(Lp.∇y, Lp.p) 
	@avx @. ẏ =  Lp.∇y[1] # ∇ⁱ ⋅ pxⁱ⋅  yx
end

# Vector field method (m==2). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{2,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}		
	m11,  m12,  m21,  m22  = Lp.mm[1,1],  Lp.mm[1,2],  Lp.mm[2,1],  Lp.mm[2,2]
	∂v11, ∂v12, ∂v21, ∂v22 = Lp.∂v[1,1], Lp.∂v[1,2], Lp.∂v[2,1], Lp.∂v[2,2]
	p1y, p2y, v1, v2       = Lp.p[1], Lp.p[2], Lp.v[1], Lp.v[2]
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
		# notice the extra y[i] at the end of the following two lines
		p1y[i]  = (m11[i] * v1[i] + m12[i] * v2[i]) * y[i]
		p2y[i]  = (m21[i] * v1[i] + m22[i] * v2[i]) * y[i]
	end
	Lp.∇!(Lp.∇y, (p1y, p2y)) # (p1y,p2y) ≡  pxⁱ⋅yx ⟹  Lp.∇y ≡ ∇ⁱ⋅pxⁱ⋅yx
	@avx @. ẏ =  Lp.∇y[1] + Lp.∇y[2]  # sumᵢ ∇ⁱ⋅pxⁱ⋅yx
end

