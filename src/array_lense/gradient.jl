# Transpose delta flow for computing the gradient with respect to the displacement field


# δᵀ Array lensing
# ===============================================
# d is the dimension of the Xfield storage 
# m is the intrinsic dimension of the field (i.e ndims(∇))

# Note that ∇! is an object that works via
#  ∇!(∇y::NTuple{m,A}, y::A) where {Tf,d, A<:Array{Tf,d}}


# δᵀArrayLense(f,v) * (δf, δv)

# δᵀArrayLense 
# --------------------------------
struct δᵀArrayLense{m,Tf,d,Tg,Tt<:Real}  <: AbstractFlow{XFields.Id{Tf,d},Tf,Tf,d}
	v::NTuple{m,Array{Tf,d}}
	∇!::Tg  
	t₀::Tt
	t₁::Tt
	nsteps::Int
	function δᵀArrayLense(v::NTuple{m,Array{Tf,d}}, ∇!::Tg, t₀::Tt, t₁::Tt, nsteps::Int) where {m,Tf,d,Tg,Tt<:Real}
		new{m,Tf,d,Tg,Tt}(v, ∇!, t₀, t₁, nsteps)
	end
end

function Base.inv(L::δᵀArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	δᵀArrayLense(L.v, L.∇!, L.t₁, L.t₀, L.nsteps)
end

# δᵀArrayLensePlan
# --------------------------------
struct δᵀArrayLensePlan{m,Tf,d,Tg,Tt<:Real}
	v::NTuple{m,Array{Tf,d}} 
	∇!::Tg   
	∂v::Matrix{Array{Tf,d}}    
	mm::Matrix{Array{Tf,d}}    
	p::NTuple{m,Array{Tf,d}}    
	∇y::NTuple{m,Array{Tf,d}}    
end

function plan(L::δᵀArrayLense{m,Tf,d,Tg,Tt}) where {m,Tf,d,Tg,Tt<:Real}
	∂v = Array{Tf,d}[zeros(Tf,size(L.v[r])) for r=1:m, c=1:m]
	for r = 1:m
		L.∇!(tuple(∂v[r,:]...), L.v[r])
	end 
	mm   = deepcopy(∂v)
	p   = deepcopy(L.v)
	∇y  = deepcopy(L.v)
	δᵀArrayLensePlan{m,Tf,d,Tg,Tt}(L.v, L.∇!, ∂v, mm, p, ∇y)
end


function (Lp::δᵀArrayLensePlan{2,Tf,d})(
		ẏ::Array{Tf,d}, 
		t::Real, 
		y::Array{Tf,d}
	) where {Tf,d,Trn<:Transform{Tf,d}}		
	m11,  m12,  m21,  m22  = Lp.mm[1,1],  Lp.mm[1,2],  Lp.mm[2,1],  Lp.mm[2,2]
	∂v11, ∂v12, ∂v21, ∂v22 = Lp.∂v[1,1], Lp.∂v[1,2], Lp.∂v[2,1], Lp.∂v[2,2]
	p1, p2, v1, v2         = Lp.p[1], Lp.p[2], Lp.v[1], Lp.v[2]
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
		p1[i]  = m11[i] * v1[i] + m12[i] * v2[i]
		p2[i]  = m21[i] * v1[i] + m22[i] * v2[i]
	end
	Lp.∇!(Lp.∇y, y) 
	@avx @. ẏ =  p1 * Lp.∇y[1] + p2 * Lp.∇y[2] # pxⁱ⋅ ∇ⁱ ⋅ yx
end



In the most general form of this we should have a linear system that flows a full 
concatonation of things ....

Normally it would be (ft...), (vt...), (δᵀft...), (δᵀvt...) but since vt doesn't depend on \
time it's just (ft..., δᵀft..., δᵀvt...)

we need (δf, δv) -> [∂(Lf,v)/∂(f,v)]⁻ᴴ* (δf, δv) 

Perhaps we start with 

function (v𝕁ᴴp::...)(
		ẏ::Y, # n is the number of fields 
		t::Real, 
		y::Y,
) where {Y <: NTuple{nm,Array{Tv,d}}} # nm = n + m = length(δᵀf) + length(δᵀv)

	# v::NTuple{m,Array{Tf,d}} 

	for r = 1:m
		v𝕁ᴴp.∇!(tuple(∂v[r,:]...), L.v[r])
	end 


end 

function update_𝒱_δᵀ_Flowϕ!(
	𝒱_δᵀ_fx::Vector{Matrix{T}}, 
	𝒱_δᵀ_ϕk::Matrix{CT}, 

	vx::Vector{Matrix{T}}, 
	px::Vector{Matrix{T}}, 
	Mx::Matrix{Matrix{T}}, 
	fx::Vector{Matrix{T}}, 
	t::T, 
	δᵀ_fx::Vector{Matrix{T}}, 
	
	write_op!::FieldFlows.Write_xk_op!{T,F}, 
	add_op!::FieldFlows.Add_xk_op!{T,F}

) where {F,T<:Real,CT<:Complex{T}}
    

    n_fields = length(δᵀ_fx)
    
    𝒱_δᵀ_ϕk .= 0
    for f ∈ 1:n_fields
        for i ∈ 1:2
            # using 𝒱_δᵀ_fx[1] as storage
            write_op!(𝒱_δᵀ_fx[f],  δᵀ_fx[f], i, fx[f])
            for j ∈ 1:2
                # ∇ᵖ ⋅ ∇⁠ᵍ ⋅ (∇ʲϕx) ⋅ Mxⁱᵖ ⋅ Mxᵍʲ ⋅ (∇ⁱfxᶠ) ⋅ δᵀ_fkᶠ 
                # ≡ ∇ᵖ ⋅ ∇⁠ᵍ ⋅ (∇ʲϕx) ⋅ Mxⁱᵖ ⋅ Mxᵍʲ ⋅ δᵀ_fxᶠ ⋅ ∇ⁱ ⋅ fxᶠ 
                add_op!(𝒱_δᵀ_ϕk, t,
                    (1,2,1,2),
                    (1,1,2,2),
                    vx[j],
                    (Mx[i,1],Mx[i,2],Mx[i,1],Mx[i,2]),
                    (Mx[1,j],Mx[1,j],Mx[2,j],Mx[2,j]),
                    𝒱_δᵀ_fx[f]
                )
            end
            # ∇ʲ ⋅ Mxⁱʲ ⋅ (∇ⁱfx) ⋅ δᵀ_fk 
            # ≡ ∇ʲ ⋅ Mxⁱʲ ⋅ δᵀ_fxᶠ ⋅ ∇ⁱ ⋅ fxᶠ
            add_op!(𝒱_δᵀ_ϕk, (1,2), (Mx[i,1],Mx[i,2]), 𝒱_δᵀ_fx[f])
        end
        # ∇ⁱ ⋅ pxⁱ ⋅ δᵀ_fxᶠ ⟶ outx
        write_op!(𝒱_δᵀ_fx[f], (1, 2), (px[1], px[2]), δᵀ_fx[f])
    end
    return nothing
end


