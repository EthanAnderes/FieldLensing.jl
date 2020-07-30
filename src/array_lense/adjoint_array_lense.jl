
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
	ArrayLenseᴴ(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

function LinearAlgebra.adjoint(L::ArrayLenseᴴ{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	ArrayLense(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
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


# Vector field method
# --------------------------------------

# (m==1). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{1,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}
	set1M!(Lp.mm[1,1], t, Lp.∂v[1,1])
	set1p!(Lp.p[1], Lp.mm[1,1], Lp.v[1])
	@avx Lp.p[1] .*= y
	div!(ẏ, Lp.∇y, Lp.p, Lp.∇!) # sumᵢ ∇ⁱ⋅pxⁱ⋅yx

end

# (m==2). Note: overwrites ẏ
function (Lp::ArrayLenseᴴPlan{2,Tf,d})(ẏ::Array{Tf,d}, t::Real, y::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}		
	
	set2M!(
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		t, 
		Lp.∂v[1,1], Lp.∂v[2,1], Lp.∂v[1,2], Lp.∂v[2,2]
	)
	set2p!(
		Lp.p[1], Lp.p[2], 
		Lp.mm[1,1],  Lp.mm[2,1],  Lp.mm[1,2],  Lp.mm[2,2], 
		Lp.v[1], Lp.v[2]
	)

	@avx Lp.p[1] .*= y
	@avx Lp.p[2] .*= y
	div!(ẏ, Lp.∇y, Lp.p, Lp.∇!) # sumᵢ ∇ⁱ⋅pxⁱ⋅yx

end



# div method which takes args (p, ∇!)
# --------------------------------------

# m == 1
function div!(
	divp::A,         # output is put in here
	∇p::NTuple{1,A}, # used for temp storage
	p::NTuple{1,A},  
	∇!::Tg, 
) where {Tg, d, Tf, A<:Array{Tf,d}}

	∇!(∇p, p) 
	@avx @. divp =  ∇p[1]

end

# m == 2
function div!(
	divp::A,         # output is put in here
	∇p::NTuple{2,A}, # used for temp storage
	p::NTuple{2,A},  
	∇!::Tg, 
) where {Tg, d, Tf, A<:Array{Tf,d}}

	∇!(∇p, p) 
	@avx @. divp =  ∇p[1] + ∇p[2] 

end
