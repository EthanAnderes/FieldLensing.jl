
# τ Array lensing
# ===============================================
# Transpose delta flow for computing the gradient with respect to the displacement field

# Note that ∇! is an object that works via
#  ∇!(∇y::NTuple{m,A}, y::A) where {Tf,d, A<:Array{Tf,d}}
# also need ∇!(∇y::NTuple{m,A}, y::NTuple{m,A})

# τL = τArrayLense(v, f, ∇!, t₀, t₁, nsteps)
# τL(δv::NTuple{m,A}, δf::NTuple{n,A}) -> (τv, τf)
# n is the number of fields that gets lensed by the same v
# m is the intrinsic dimension of the space
# d is the Array dimension that forms the storage container for the fields f 

# τArrayLense 
# --------------------------------
struct τArrayLense{m,n,Tf,d,Tg,Tt<:Real} 
	v::NTuple{m,Array{Tf,d}}
	f::NTuple{n,Array{Tf,d}} # defined at t₀
	∇!::Tg  
	t₀::Tt
	t₁::Tt
	nsteps::Int
	function τArrayLense(v::NTuple{m,A}, f::NTuple{n,A}, ∇!::Tg, t₀::Tt, t₁::Tt, nsteps::Int) where {m,n,Tf,d,A<:Array{Tf,d},Tg,Tt<:Real}
		new{m,n,Tf,d,Tg,Tt}(v, f, ∇!, t₀, t₁, nsteps)
	end
end


# τArrayLense operating on (τf, τv)
# --------------------------------

function (τL::τArrayLense{m,n,Tf,d,Tg,Tt})(
		τv::NTuple{m,A}, 
		τf::NTuple{n,A}
	)::Tuple{NTuple{m,A}, NTuple{n,A}} where {m,n,Tf,d,Tg,Tt<:Real,A<:Array{Tf,d}}
	
	pτL!  = plan(τL) 
	rtn   = odesolve_RK4(pτL!, tuple(τv..., τf..., τL.f...), τL.t₀, τL.t₁, τL.nsteps)
	return tuple(rtn[Base.OneTo(m)]...), tuple(rtn[(m+1):(m+n)]...)

end


# *, \, inv τArrayLense, need to move τL.f from time t₀ to time t₁
# --------------------------------

function Base.:*(τL::τArrayLense{m,n,Tf,d}, τvτf::Tuple{NTuple{m,A}, NTuple{n,A}}) where {m,n,Tf,d,A<:Array{Tf,d}}
	τL(τvτf[1], τvτf[2])
end

function Base.:\(τL::τArrayLense{m,n,Tf,d}, τvτf::Tuple{NTuple{m,A}, NTuple{n,A}}) where {m,n,Tf,d,A<:Array{Tf,d}}
	invτL = inv(τL)
	invτL(τvτf[1], τvτf[2])
end

function Base.inv(τL::τArrayLense{m,n,Tf,d,Tg,Tt}) where {m,n,Tf,d,Tg,Tt<:Real}
	L = ArrayLense(τL.v, τL.∇!, τL.t₀, τL.t₁, τL.nsteps)
	ft₁ = map(f -> flow(L,f), τL.f)
	τArrayLense(τL.v, ft₁, τL.∇!, τL.t₁, τL.t₀, τL.nsteps)
end

# τArrayLensePlan (note: the Plan doesn't hold the tuple of fields but it knows how many there are)
# --------------------------------
struct τArrayLensePlan{m,n,Tf,d,Tg,Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	w::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}} # for storage 
	∇x::NTuple{m,Array{Tf,d}} # for storage  
end

function plan(L::τArrayLense{m,n,Tf,d,Tg,Tt}) where {m,n,Tf,d,Tg,Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm  = deepcopy(∂v)
	p   = deepcopy(L.v)
	w   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	∇x  = deepcopy(L.v)
	τArrayLensePlan{m,n,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, w, ∇y, ∇x)
end



# ODE vector field τLp(ẏ, t, y) where τLp::τArrayLensePlan 
# --------------------------------

# m == 2 case
function (τLp::τArrayLensePlan{2,n,Tf,d})(ẏ, t, y) where {n,Tf,d}
	# we need updated M and p for current time t
	# --------------------------
	setpM!(
		τLp.p[1], τLp.p[2], 
		τLp.mm[1,1],  τLp.mm[2,1],  τLp.mm[1,2],  τLp.mm[2,2], 
		t, 
		τLp.v[1], τLp.v[2], 
		τLp.∂v[1,1], τLp.∂v[2,1], τLp.∂v[1,2], τLp.∂v[2,2]
	)

	# update τ̇f and ḟ (and τLp.w for use in updating τ̇v)
	# --------------------------
	# initialize τLp.w (it will get updated in the following loop) 
	@avx @. τLp.w[1] = 0
	@avx @. τLp.w[2] = 0
	for i = 1:n
		τ̇f, τf = ẏ[2+i],   y[2+i] 	# ẏ[m+i],   y[m+i]
		ḟ, f   = ẏ[2+n+i], y[2+n+i] # ẏ[m+n+i], y[m+n+i]   

		# fill τ̇f = ∇ⁱpⁱτf (make sure τLp.p is pre-computed)
		∇ⁱvⁱf!(τ̇f, τLp.p, τf, τLp.∇!, τLp.∇x, τLp.∇y)

		# fill ḟ (save ∇f in τLp.∇y for storage)
		τLp.∇!(τLp.∇y, f)  
		@avx @. ḟ =  τLp.p[1] * τLp.∇y[1] + τLp.p[2] * τLp.∇y[2] # pxⁱ⋅ ∇ⁱ ⋅ yx

		# add 2 w (use ∇f stored in τLp.∇y)
		# τf * Mᴴ * ∇f, note the transpose on M
		@avx @. τLp.w[1] += τf * (τLp.mm[1,1] * τLp.∇y[1] + τLp.mm[2,1] * τLp.∇y[2])  
		@avx @. τLp.w[2] += τf * (τLp.mm[1,2] * τLp.∇y[1] + τLp.mm[2,2] * τLp.∇y[2]) 
	end

	# update τ̇v ≡ ẏ[1:m]
	# ----------------------
	for i = 1:2 # m == 2
		@avx @. τLp.∇x[1] = τLp.p[1]*τLp.w[i]  
		@avx @. τLp.∇x[2] = τLp.p[2]*τLp.w[i]  
		τLp.∇!(τLp.∇y, τLp.∇x)  
		@avx @. ẏ[i] = - τLp.w[i] - t * (τLp.∇y[1] + τLp.∇y[2])
	end

end
