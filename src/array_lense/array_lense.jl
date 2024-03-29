
# Array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Basic methods used in most ArrayLense methods
# --------------------------------------

function ∇ⁱvⁱ!(s::A, v::NTuple{m,A}, ∇!, ∇x::NTuple{m,A}) where {m,A<:AbstractMatrix}
    ∇!(∇x, v)  
    @inbounds @. s = ∇x[1]
    for i = 2:m
        @inbounds @. s += ∇x[i]
    end
    return s
end

function ∇ⁱvⁱf!(s::A, v::NTuple{m,A}, f::A, ∇!, ∇x::NTuple{m,A}, ∇y::NTuple{m,A}) where {m,A<:AbstractMatrix}
    for i = 1:m
        @inbounds @. ∇x[i] = v[i] * f
    end
    ∇ⁱvⁱ!(s, ∇x, ∇!, ∇y)
end


function vⁱ∇ⁱf!(s::A, v::NTuple{m,A}, f::A, ∇!, ∇x::NTuple{m,A}) where {m,A<:AbstractMatrix}
    ∇!(∇x, f)
    @inbounds @. s = ∇x[1] * v[1]
    for i = 2:m
        @inbounds @. s += ∇x[i] * v[i]
    end
    return s
end

# just to make these as fast as possible for m==2 we can specialize
# -------------------------------------------------------


function ∇ⁱvⁱ!(s::A, v::NTuple{2,A}, ∇!, ∇x::NTuple{2,A}) where {A<:AbstractMatrix}
    ∇!(∇x, v)  
    @inbounds @. s = ∇x[1] + ∇x[2]
    return s
end

function vⁱ∇ⁱf!(s::A, v::NTuple{2,A}, f::A, ∇!, ∇x::NTuple{2,A}) where {A<:AbstractMatrix}
    ∇!(∇x, f)
    @inbounds @. s = v[1] * ∇x[1] + v[2] * ∇x[2] 
    return s
end

# version 1
function ∇ⁱvⁱf!(s::A, v::NTuple{2,A}, f::A, ∇!, ∇x::NTuple{2,A}, ∇y::NTuple{2,A}) where {A<:AbstractMatrix}
   @inbounds @. ∇x[1] = v[1] * f
   @inbounds @. ∇x[2] = v[2] * f
   ∇ⁱvⁱ!(s, ∇x, ∇!, ∇y)
end
# version 2 (this first distributes the derivative onto the product)
## function ∇ⁱvⁱf!(s::A, v::NTuple{2,A}, f::A, ∇!, ∇x::NTuple{2,A}, ∇y::NTuple{2,A}) where {A<:AbstractMatrix}
##     ∇!(∇x, v)
##     ∇!(∇y, f)
##     @turbo @. s = (∇x[1] + ∇x[2]) * f + v[1] * ∇y[1] + v[2] * ∇y[2]
##     return s
## end


# ArrayLense
# --------------------------------
struct ArrayLense{m,Tf,d,Tg<:Gradient{m},Tt<:Real}  <: AbstractFlow
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
struct ArrayLensePlan{m,Tf,d,Tg<:Gradient{m},Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}    
end

function plan(L::ArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg<:Gradient{m},Tt<:Real}
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
function (Lp::ArrayLensePlan{1,Tf,d})(ẏ::A, t::Real, y::A) where {Tf,d,A<:Array{Tf,d}}
	setpM!(Lp.p[1], Lp.mm[1,1], t, Lp.v[1], Lp.∂v[1,1])
	vⁱ∇ⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
end

# for the case when lensing multiple fields
function (Lp::ArrayLensePlan{1,Tf,d})(ẏ::NTuple{n,A}, t::Real, y::NTuple{n,A}) where {n,Tf,d,A<:Array{Tf,d}}
	setpM!(Lp.p[1], Lp.mm[1,1], t, Lp.v[1], Lp.∂v[1,1])
	for i=1:n
		vⁱ∇ⁱf!(ẏ[i], Lp.p, y[i], Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
	end
end

# (m==2). Note: overwrites ẏ
function (Lp::ArrayLensePlan{2,Tf,d})(ẏ::A, t::Real, y::A) where {Tf,d,A<:Array{Tf,d}}	
	setpM!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.v[1], Lp.v[2], 
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	vⁱ∇ⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
end

# for the case when lensing multiple fields
function (Lp::ArrayLensePlan{2,Tf,d})(ẏ::NTuple{n,A}, t::Real, y::NTuple{n,A})  where {n,Tf,d,A<:Array{Tf,d}}	
	setpM!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.v[1], Lp.v[2], 
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	for i=1:n
		vⁱ∇ⁱf!(ẏ[i], Lp.p, y[i], Lp.∇!, Lp.∇y) # ẏ = pⁱ∇ⁱy
	end
end


# Methods that fill M and p from (v,∂v,t) 
# --------------------------------------

# m == 1
function setpM!(m, t, v, ∂v)
	@inbounds @. m  = 1 / (1 + t * ∂v)
	@inbounds @. p  = m * v
end

# m == 2
function setpM!(p1, p2, m11,  m21,  m12,  m22, t, v1, v2, ∂v11, ∂v21, ∂v12, ∂v22)
	# @tturbo for i ∈ eachindex(m11)
	Base.Threads.@threads for i ∈ eachindex(m11)
		m11[i] = 1 + t * ∂v22[i] 
		m12[i] =   - t * ∂v12[i] 
		m21[i] =   - t * ∂v21[i] 
		m22[i] = 1 + t * ∂v11[i] 
		dt     = m11[i] * m22[i] - m12[i] * m21[i]
		m11[i] /= dt 
		m12[i] /= dt
		m21[i] /= dt
		m22[i] /= dt
		p1[i]  = m11[i] * v1[i] + m12[i] * v2[i]
 		p2[i]  = m21[i] * v1[i] + m22[i] * v2[i]
	end
end





