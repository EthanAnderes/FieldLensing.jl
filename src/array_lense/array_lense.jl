
# Array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Basic methods used in most ArrayLense methods
# --------------------------------------
# Need to define a struct with an instance 
# that can operate on arguments as follows 
#  ∇!(∇y::NTuple{m,A}, f::A)           ->  ∇y = (∇¹f,∇²f, ..., ∇ᵐf)
#  ∇!(∇y::NTuple{m,A}, v::NTuple{m,A}) ->  ∇y = (∇¹v¹,∇²v², ..., ∇ᵐvᵐ)

function ∇ⁱvⁱ!(s::A, v::NTuple{m,A}, ∇!, ∇x::NTuple{m,A}) where {m,A<:AbstractMatrix}
    ∇!(∇x, v)  
    @avx @. s = ∇x[1]
    for i = 2:m
        @avx @. s += ∇x[i]
    end
    return s
end

function ∇ⁱvⁱf!(s::A, v::NTuple{m,A}, f::A, ∇!, ∇x::NTuple{m,A}, ∇y::NTuple{m,A}) where {m,A<:AbstractMatrix}
    for i = 1:m
        @avx @. ∇y[i] = v[i] * f
    end
    ∇ⁱvⁱ!(s, ∇y, ∇!, ∇x)
end

function vⁱ∇ⁱf!(s::A, v::NTuple{m,A}, f::A, ∇!, ∇x::NTuple{m,A}) where {m,A<:AbstractMatrix}
    ∇!(∇x, f)
    @avx @. s = ∇x[1] * v[1]
    for i = 2:m
        @avx @. s += ∇x[i] * v[i]
    end
    return s
end

# just to make these as fast as possible for m==2 we can specialize

function ∇ⁱvⁱ!(s::A, v::NTuple{2,A}, ∇!, ∇x::NTuple{2,A}) where {A<:AbstractMatrix}
    ∇!(∇x, v)  
    @avx @. s = ∇x[1] + ∇x[2]
    return s
end

function vⁱ∇ⁱf!(s::A, v::NTuple{2,A}, f::A, ∇!, ∇x::NTuple{2,A}) where {A<:AbstractMatrix}
    ∇!(∇x, f)
    @avx @. s = ∇x[1] * v[1] + ∇x[2] * v[2]
    return s
end


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



# ArrayLensePlan's generate the vector field methods
# --------------------------------------

# (m==1). Note: overwrites ẏ
function (Lp::ArrayLensePlan{1,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}
	setpM!(Lp.p[1], Lp.mm[1,1], t, Lp.v[1], Lp.∂v[1,1])
	vⁱ∇ⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
end


# (m==2). Note: overwrites ẏ
function (Lp::ArrayLensePlan{2,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}		
	setpM!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.v[1], Lp.v[2], 
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	vⁱ∇ⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
end


# Methods that fill M and p from (v,∂v,t) 
# --------------------------------------

# m == 1
function setpM!(m, t, v, ∂v)
	@avx @. m  = 1 / (1 + t * ∂v)
	@avx @. p  = m * v
end


# m == 2
function setpM!(p1, p2, m11,  m21,  m12,  m22, t, v1, v2, ∂v11, ∂v21, ∂v12, ∂v22)
	@avx for i ∈ eachindex(m11)
		m11[i] = 1 + t * ∂v22[i] 
		m12[i] =   - t * ∂v12[i] 
		m21[i] =   - t * ∂v21[i] 
		m22[i] = 1 + t * ∂v11[i] 
		dt     = m11[i] * m22[i] - m12[i] * m21[i]
		p1[i]  = (m11[i] * v1[i] + m12[i] * v2[i])/dt
 		p2[i]  = (m21[i] * v1[i] + m22[i] * v2[i])/dt
		m11[i] /= dt
		m12[i] /= dt
		m21[i] /= dt
		m22[i] /= dt
	end
end




