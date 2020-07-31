# Transpose delta flow for computing the gradient with respect to the displacement field


# τ Array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Note that ∇! is an object that works via
#  ∇!(∇y::NTuple{m,A}, y::A) where {Tf,d, A<:Array{Tf,d}}


# τL = τArrayLense(v, ∇!, t₀, t₁, nsteps)
# τL(f::A, δf::A, δv::NTuple{m,A}) -> (f, τf, τv)

# τArrayLense 
# --------------------------------
struct τArrayLense{m,Tf,d,Tg,Tt<:Real} # <: AbstractFlow{XFields.Id{Tf,d},Tf,Tf,d}
	v::NTuple{m,Array{Tf,d}}
	∇!::Tg  
	t₀::Tt
	t₁::Tt
	nsteps::Int
	function τArrayLense(v::NTuple{m,Array{Tf,d}}, ∇!::Tg, t₀::Tt, t₁::Tt, nsteps::Int) where {m,Tf,d,Tg,Tt<:Real}
		new{m,Tf,d,Tg,Tt}(v, ∇!, t₀, t₁, nsteps)
	end
end

function (τL::τArrayLense{m,Tf,d,Tg,Tt})(f::A, τf::A, τv::NTuple{m,A}) where {m,Tf,d,Tg,Tt<:Real,A<:Array{Tf,d}}
	# pack f, τf, τv into Array{Tf,d+1}
	fτfτv  = cat(f, τf, τv...; dims = d+1)::Array{Tf,d+1}
	pτL!   = plan(τL) 
	fτfτv′ = odesolve_RK4(pτL!, fτfτv, τL.t₀, τL.t₁, τL.nsteps)
	# unpack
	rtn_f  = A(selectdim(fτfτv′, d+1, 1))
	rtn_τf = A(selectdim(fτfτv′, d+1, 2))
	rtn_τv = tuple((A(selectdim(fτfτv′, d+1, i+2)) for i = Base.OneTo(m))...)
	return rtn_f, rtn_τf, rtn_τv
end

function Base.inv(L::τArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	τArrayLense(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

# τArrayLensePlan
# --------------------------------
struct τArrayLensePlan{m,Tf,d,Tg,Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	w::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}   # for storage 
	∇x::NTuple{m,Array{Tf,d}}   # for storage  
end

function plan(L::τArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm  = deepcopy(∂v)
	p   = deepcopy(L.v)
	w   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	∇x  = deepcopy(L.v)
	τArrayLensePlan{m,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, w, ∇y, ∇x)
end

# m is the dimension of the space, gives NTuple length
# d is the storage dimension of a single field or vector field coordinate 
# d′ = d + 1, used to stack all fields into an Array for use by ode_solvers
function (τLp::τArrayLensePlan{2,Tf,d})(
		ẏ::A, # adding an extra dimension to hold everything
		t::Real, 
		y::A
	) where {Tf,d′,d, A<:Array{Tf,d′}, Trn<:Transform{Tf,d}}		

	@assert d′ == d + 1

	set2M!(
		τLp.mm[1,1],  τLp.mm[2,1],  τLp.mm[1,2],  τLp.mm[2,2], 
		t, 
		τLp.∂v[1,1], τLp.∂v[2,1], τLp.∂v[1,2], τLp.∂v[2,2]
	)
	set2p!(
		τLp.p[1], τLp.p[2], 
		τLp.mm[1,1],  τLp.mm[2,1],  τLp.mm[1,2],  τLp.mm[2,2], 
		τLp.v[1], τLp.v[2]
	)

	# unpack input arrays
	# --------------------------
	# FIXME: how to get type inference here?
	S  = SubArray{Tf,2,A}
	f, τf    = selectdim(y, d′, 1)::S, selectdim(y, d′, 2)::S
	ḟ, τ̇f    = selectdim(ẏ, d′, 1)::S, selectdim(ẏ, d′, 2)::S
	τv1, τv2 = selectdim(y, d′, 3)::S, selectdim(y, d′, 4)::S
	τ̇v1, τ̇v2 = selectdim(ẏ, d′, 3)::S, selectdim(ẏ, d′, 4)::S

	# fill τ̇f (use τLp.∇y for storage)
	# --------------------------
	@avx @. τLp.∇x[1] = τLp.p[1] * τf
	@avx @. τLp.∇x[2] = τLp.p[2] * τf
	τLp.∇!(τLp.∇y, τLp.∇x) # stored in τLp.∇y
	@avx @. τ̇f =  τLp.∇y[1] + τLp.∇y[2] 
	
	# fill ḟ (save ∇f in τLp.∇y for storage)
	# --------------------------
	
	τLp.∇!(τLp.∇y, f)  
	@avx @. ḟ =  τLp.p[1] * τLp.∇y[1] + τLp.p[2] * τLp.∇y[2] # pxⁱ⋅ ∇ⁱ ⋅ yx


	# fill τ̇v (use ∇f stored in τLp.∇y)
	# --------------------------

	# compute w by hijacking p constructor 
	set2p!(
		τLp.w[1], τLp.w[2], 
		τLp.mm[1,1],  τLp.mm[1,2], τLp.mm[2,1], τLp.mm[2,2], #<- note the mm transpose
		τLp.∇y[1], τLp.∇y[2] # currently holding ∇f
	)
	# compute w, then multiply by - τf (still store in w)
	@avx @. τLp.w[1] *= - τf
	@avx @. τLp.w[2] *= - τf

	# set initial τ̇v to `- w * τf`
	@avx @. τ̇v1 = τLp.w[1] 
	@avx @. τ̇v2 = τLp.w[2] 

	# Note: τLp.w is technically `- w * τf` at this point
	# now add ∂₁ * w1 * p + ∂₂ * w2 * p
	# w1 * p = (w[1] * p[1], w[1] * p[2]) 
	# w2 * p = (w[2] * p[1], w[2] * p[2]) 
	## by swapping coordinates we can re-use Nabla! 	

	@avx @. τLp.∇x[1] = τLp.w[1] * τLp.p[1]
	@avx @. τLp.∇x[2] = τLp.w[2] * τLp.p[1]
	τLp.∇!(τLp.∇y, τLp.∇x)
	@avx @. τ̇v1 += τLp.∇y[1] 
	@avx @. τ̇v1 += τLp.∇y[2] 

	@avx @. τLp.∇x[1] = τLp.w[1] * τLp.p[2]
	@avx @. τLp.∇x[2] = τLp.w[2] * τLp.p[2]	
 	τLp.∇!(τLp.∇y, τLp.∇x)
	@avx @. τ̇v2 += τLp.∇y[1] 
	@avx @. τ̇v2 += τLp.∇y[2] 

end



