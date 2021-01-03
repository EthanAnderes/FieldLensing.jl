
# Adjoint array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# TODO: perhaps we want the user to explicitly define
# the adjoint of the partial derivatives and then use them 
# in the transpose lensing ...

# ArrayLenseᴴ
# --------------------------------
struct ArrayLenseᴴ{m,Tf,d,Tg<:Gradient{m},Tt<:Real}  <: AbstractFlow{XFields.Id{Tf,d},Tf,Tf,d}
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
	ArrayLenseᴴ(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLense(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end


# ArrayLenseᴴPlan
# --------------------------------
struct ArrayLenseᴴPlan{m,Tf,d,Tg<:Gradient{m},Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}    
	∇x::NTuple{m,Array{Tf,d}}    
end

function plan(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg<:Gradient{m},Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm  = deepcopy(∂v)
	p   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	∇x  = deepcopy(L.v)
	ArrayLenseᴴPlan{m,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, ∇y, ∇x)
end


# Vector field method
# --------------------------------------

# (m==1). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{1,Tf,d})(ẏ::A, t::Real, y::A) where {Tf,d,A<:Array{Tf,d}}
	setpM!(Lp.p[1], Lp.mm[1,1], t, Lp.v[1], Lp.∂v[1,1])
	∇ⁱvⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇x, Lp.∇y) # ∇ⁱpⁱy
end

# for the case when lensing multiple fields
function (Lp::ArrayLenseᴴPlan{1,Tf,d})(ẏ::NTuple{n,A}, t::Real, y::NTuple{n,A}) where {n,Tf,d,A<:Array{Tf,d}}
	setpM!(Lp.p[1], Lp.mm[1,1], t, Lp.v[1], Lp.∂v[1,1])
	for i=1:n
		∇ⁱvⁱf!(ẏ[i], Lp.p, y[i], Lp.∇!, Lp.∇x, Lp.∇y) # ∇ⁱpⁱy
	end
end


# (m==2). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{2,Tf,d})(ẏ::A, t::Real, y::A) where {Tf,d,A<:Array{Tf,d}}
	setpM!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.v[1], Lp.v[2],
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	∇ⁱvⁱf!(ẏ, Lp.p, y, Lp.∇!, Lp.∇x, Lp.∇y) # ∇ⁱpⁱy
end

# for the case when lensing multiple fields
function (Lp::ArrayLenseᴴPlan{2,Tf,d})(ẏ::NTuple{n,A}, t::Real, y::NTuple{n,A}) where {n,Tf,d,A<:Array{Tf,d}}
	setpM!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.v[1], Lp.v[2],
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	for i=1:n
		∇ⁱvⁱf!(ẏ[i], Lp.p, y[i], Lp.∇!, Lp.∇x, Lp.∇y) # ∇ⁱpⁱy
	end
end



