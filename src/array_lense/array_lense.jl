
# Array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Need to define a struct with an instance 
# that can operate on arguments as follows 
#  ∇!(∇y::NTuple{m,A}, y::A) where {Tf,d, A<:Array{Tf,d}}

# ArrayLense
# --------------------------------
struct ArrayLense{m,Tf,d,Tg,Tt<:Real}  <: AbstractFlow{XFields.Id{Tf,d},Tf,Tf,d}
	v::NTuple{m,Array{Tf,d}}
	∇!::Tg  
	t₀::Tt
	t₁::Tt
	nsteps::Int
	function ArrayLense(v::NTuple{m,Array{Tf,d}}, ∇!::Tg, t₀::Tt, t₁::Tt, nsteps::Int) where {m,Tf,d,Tg,Tt<:Real}
		new{m,Tf,d,Tg,Tt}(v, ∇!, t₀, t₁, nsteps)
	end
end

function Base.inv(L::ArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLense(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

# ArrayLensePlan
# --------------------------------
struct ArrayLensePlan{m,Tf,d,Tg,Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}    
end

function plan(L::ArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm   = deepcopy(∂v)
	p   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	ArrayLensePlan{m,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, ∇y)
end

# Vector field method
# --------------------------------------

# (m==1). Note: overwrites ẏ
function (Lp::ArrayLensePlan{1,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}
	setMp!(Lp.mm, Lp.p, Lp.∂v, Lp.v, t)
	Lp.∇!(Lp.∇y, y) 
	@avx @. ẏ =  Lp.p[1] * Lp.∇y[1] # pxⁱ⋅ ∇ⁱ ⋅ yx
end

# (m==2). Note: overwrites ẏ
function (Lp::ArrayLensePlan{2,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}		
	setMp!(Lp.mm, Lp.p, Lp.∂v, Lp.v, t)
	Lp.∇!(Lp.∇y, y) 
	@avx @. ẏ =  Lp.p[1] * Lp.∇y[1] + Lp.p[2] * Lp.∇y[2] # pxⁱ⋅ ∇ⁱ ⋅ yx
end


# Method that fills (M, p) from (v,∂v) 
# --------------------------------------

# m == 1, i.e. intrinsic dimension is 1
function setMp!(
	mm::Matrix{A}, p::NTuple{1,A}, # <- mm and p are modified   
	∂v::Matrix{A}, v::NTuple{1,A}, 
	t::Real,
) where {d, Tf, A<:Array{Tf,d}}

	@avx @. mm[1,1]  = 1 / (1 + t * ∂v[1,1])
	@avx @. p[1]     = mm[1,1] * v[1]

end

# m == 2, i.e. intrinsic dimension is 2
function setMp!(
	mm::Matrix{A}, p::NTuple{2,A}, # <- mm and p are modified   
	∂v::Matrix{A}, v::NTuple{2,A}, 
	t::Real,
) where {d, Tf, A<:Array{Tf,d}}

	m11,  m12,  m21,  m22  = mm[1,1],  mm[1,2],  mm[2,1],  mm[2,2]
	∂v11, ∂v12, ∂v21, ∂v22 = ∂v[1,1], ∂v[1,2], ∂v[2,1], ∂v[2,2]
	p1, p2, v1, v2         = p[1], p[2], v[1], v[2]

	@avx for i ∈ eachindex(m11)
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

end

