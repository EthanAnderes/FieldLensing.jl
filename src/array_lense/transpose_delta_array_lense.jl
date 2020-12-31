
# τ Array lensing
# ===============================================
# Transpose delta flow for computing the gradient 
# with respect to the displacement field.

# τL = τArrayLense(v, f, ∇!, t₀, t₁, nsteps)
# τL(δv::NTuple{m,A}, δf::NTuple{n,A}) -> (τv, τf)

# n is the number of fields that gets lensed by the same v
# m is the intrinsic dimension of the space
# d is the Array dimension that forms the storage container for the fields f 


# Highlevel transpose delta flow functions 
# --------------------------------

"""
ᴴ∂Ł⁻¹fx_∂vx(v,f,g,...) -> [∂Ł⁻¹fx_∂vx]ᴴ * g, transpose delta flow from time 0 to 1 

Inputs, are (gx::NTuple{n,A}, fx::NTuple{n,A}, vx::NTuple{m,A}, ∇!::Tg, grad_nsteps::Int)
where fx and gx should be tuples of arrays defined at time 0. 
"""
function ᴴ∂Ł⁻¹fx_∂vx(gx::NTuple{n,A}, fx::NTuple{n,A}, vx::NTuple{m,A}, ∇!::Tg, grad_nsteps::Int)::NTuple{m,A} where {m,n,Tf,d,A<:Array{Tf,d},Tg}
    τŁ₀₁     = τArrayLense(vx, fx, ∇!, 0, 1, grad_nsteps)
    τvx, τgx = τŁ₀₁(map(zero,vx),  gx)
    return τvx
end

"""
ᴴ∂Łfx_∂vx(v,f,g,...) -> [∂Łfx_∂vx]ᴴ * g, transpose delta flow from time 1 to 0 

Inputs, are (gx::NTuple{n,A}, fx::NTuple{n,A}, vx::NTuple{m,A}, ∇!::Tg, grad_nsteps::Int)
where fx and gx should be tuples of arrays defined at time 0. 
"""
function ᴴ∂Łfx_∂vx(gx::NTuple{n,A}, fx::NTuple{n,A}, vx::NTuple{m,A}, ∇!::Tg, grad_nsteps::Int)::NTuple{m,A} where {m,n,Tf,d,A<:Array{Tf,d},Tg}
    τŁ₁₀     = τArrayLense(vx, fx, ∇!, 1, 0, grad_nsteps)
    τvx, τgx = τŁ₁₀(map(zero,vx),  gx)
    return τvx
end



# τArrayLense type which also operates on (τf, τv)
# --------------------------------
struct τArrayLense{m,n,Tf,d,Tg<:Gradient{m},Tt<:Real} 
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
struct τArrayLensePlan{m,n,Tf,d,Tg<:Gradient{m},Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	w::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}} # for storage 
	∇x::NTuple{m,Array{Tf,d}} # for storage  
end

function plan(L::τArrayLense{m,n,Tf,d,Tg,Tt}) where {m,n,Tf,d,Tg<:Gradient{m},Tt<:Real}
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
	setpM!(
		τLp.p[1], τLp.p[2], 
		τLp.mm[1,1],  τLp.mm[2,1],  τLp.mm[1,2],  τLp.mm[2,2], 
		t, 
		τLp.v[1], τLp.v[2], 
		τLp.∂v[1,1], τLp.∂v[2,1], τLp.∂v[1,2], τLp.∂v[2,2]
	)

	# update τ̇f and ḟ (and τLp.w for use in updating τ̇v)
	# initialize τLp.w (it will get updated in the following loop) 
	τLp.w[1] .= 0
	τLp.w[2] .= 0
	for i = 1:n
		τ̇f, τf = ẏ[2+i],   y[2+i] 	# ẏ[m+i],   y[m+i]
		ḟ, f   = ẏ[2+n+i], y[2+n+i] # ẏ[m+n+i], y[m+n+i]   

		# fill τ̇f = ∇ⁱpⁱτf (make sure τLp.p is pre-computed)
		# ------------
		∇ⁱvⁱf!(τ̇f, τLp.p, τf, τLp.∇!, τLp.∇x, τLp.∇y)

		# fill ḟ = pⁱ∇ⁱf (save ∇f in τLp.∇y for storage)
		# Note: make sure τLp.∇y is overwriten with ∇f
		# ------------
		vⁱ∇ⁱf!(ḟ , τLp.p, f, τLp.∇!, τLp.∇y) # need τLp.∇y here .. not τLp.∇x
		# ---- alt 
		## τLp.∇!(τLp.∇y, f)  
		## @avx @. ḟ =  τLp.p[1] * τLp.∇y[1] + τLp.p[2] * τLp.∇y[2] 

		# Add to w (use ∇f stored in τLp.∇y)
		# τf * Mᴴ * ∇f, note the transpose on M
		
		# ------------
		## @inbounds @. τLp.∇y[1] *= τf 
		## @inbounds @. τLp.∇y[2] *= τf 
		## @avx @. τLp.w[1] += τLp.mm[1,1] * τLp.∇y[1] + τLp.mm[2,1] * τLp.∇y[2]  
		## @avx @. τLp.w[2] += τLp.mm[1,2] * τLp.∇y[1] + τLp.mm[2,2] * τLp.∇y[2] 
		## --- alt option
		## @avx @. τLp.w[1] += τf * (τLp.mm[1,1] * τLp.∇y[1] + τLp.mm[2,1] * τLp.∇y[2])  
		## @avx @. τLp.w[2] += τf * (τLp.mm[1,2] * τLp.∇y[1] + τLp.mm[2,2] * τLp.∇y[2]) 
		## --- alt option
		Base.Threads.@threads for ii ∈ eachindex(τLp.w[1])
			@inbounds τLp.w[1][ii] += τf[ii] * (τLp.mm[1,1][ii] * τLp.∇y[1][ii] + τLp.mm[2,1][ii] * τLp.∇y[2][ii])  
			@inbounds τLp.w[2][ii] += τf[ii] * (τLp.mm[1,2][ii] * τLp.∇y[1][ii] + τLp.mm[2,2][ii] * τLp.∇y[2][ii]) 
		end
	end

	# update ẏ[1:m] ≡ τ̇v
	# ----------------------
	# fill τ̇v = - w[q] - t * ∇ⁱpⁱw[q] (make sure τLp.p is pre-computed)
	# since we don't need τLp.mm at this point lets use it for temp storage
	∇ⁱvⁱf!(τLp.mm[1,1], τLp.p, τLp.w[1], τLp.∇!, τLp.∇x, τLp.∇y)
	∇ⁱvⁱf!(τLp.mm[2,1], τLp.p, τLp.w[2], τLp.∇!, τLp.∇x, τLp.∇y)
	
	## --- 
	# @avx @. ẏ[1] =  - τLp.w[1] - t * τLp.mm[1,1]
	# @avx @. ẏ[2] =  - τLp.w[2] - t * τLp.mm[2,1]
	## --- alt option
	Base.Threads.@threads for ii ∈ eachindex(ẏ[1])
		@inbounds ẏ[1][ii] =  - τLp.w[1][ii] - t * τLp.mm[1,1][ii]
		@inbounds ẏ[2][ii] =  - τLp.w[2][ii] - t * τLp.mm[2,1][ii]
	end

	# alt option for update ẏ[1:m] ≡ τ̇v
	# ----------------------
	## for i = 1:2 # m == 2
	## 	@avx @. τLp.∇x[1] = τLp.p[1] * τLp.w[i]  
	## 	@avx @. τLp.∇x[2] = τLp.p[2] * τLp.w[i]  
	## 	τLp.∇!(τLp.∇y, τLp.∇x)  
	## 	@avx @. ẏ[i] = - τLp.w[i] - t * (τLp.∇y[1] + τLp.∇y[2])
	## end

end
