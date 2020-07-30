using FieldLensing
using Test
using Spectra
using XFields
using FFTransforms
using LBblocks
using SparseArrays
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using LoopVectorization

@testset "Examples" begin


	# To use ArrayLense we just need to define ∇!
	# -----------------------------------------------
	struct Nabla!{Tθ,Tφ}
	    ∂θ::Tθ
	    ∂φᵀ::Tφ
	end

	function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
	    mul!(∇y[1], ∇!.∂θ, y[1])
	    mul!(∇y[2], y[2], ∇!.∂φᵀ)
	    ∇y
	end
	
	function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
	    ∇!(∇y, (y,y))
	end
	
	function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
	    ∇y = (similar(y), similar(y))
	    ∇!(∇y, (y,y))
	    ∇y
	end

	struct Jacobian!{Tθ,Tφ}
	    ∂θ::Tθ
	    ∂φᵀ::Tφ
	end

	function (𝕁!::Jacobian!{Tθ,Tφ})(y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
		y11, y21, y12, y22 = similar(y[1]), similar(y[1]), similar(y[1]), similar(y[1])
		mul!(y11, ∇!.∂θ, y[1])
		mul!(y21, ∇!.∂θ, y[2])
		mul!(y12, y[1], ∇!.∂φᵀ)
		mul!(y22, y[2], ∇!.∂φᵀ)
		y11, y21, y12, y11
	end

	# -----------------------------------------------
	trm, ∇!, 𝕁! = @sblock let Δθ′ = 2.5, Δφ′ = 1.0, nθ = 4*512, nφ = 5*512 - 1
    	#𝕨      = r𝕎32(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
    	𝕨      = r𝕎(nθ, nθ * deg2rad(Δθ′/60)) ⊗ 𝕎(nφ, nφ * deg2rad(Δφ′/60))
    	trm    = ordinary_scale(𝕨)*𝕨

    	onesnθm1 = fill(1,nθ-1)
    	∂θ = spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    	# ∂θ[1,:] .= 0
    	# ∂θ[end,:] .= 0
    	∂θ[1,end] = -1
	    ∂θ[end,1] =  1
	    ∂θ = (1 / (2 * Δpix(trm)[1])) * ∂θ

	    onesnφm1 = fill(1,nφ-1)
	    ∂φ      = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
	    ## for the periodic boundary conditions
	    ∂φ[1,end] = -1
	    ∂φ[end,1] =  1
	    ## now as a right operator
	    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
	    ∂φᵀ = transpose((1 / (2*Δpix(trm)[2])) * ∂φ);

	    ∇! = Nabla!(∂θ, ∂φᵀ)
	    𝕁! = Jacobian!(∂θ, ∂φᵀ)

	    return trm, ∇!, 𝕁!
	end


	# ------------------------
	v, t = @sblock let trm, ∇!, scale_lense = 1.25
    	l   = wavenum(trm)
	    cTl = Spectra.cTl_besselj_approx.(l)
    	cϕl = Spectra.cϕl_approx.(l) 

    	Ct  = DiagOp(Xfourier(trm, cTl)) 
    	Cϕ  = scale_lense * DiagOp(Xfourier(trm, cϕl)) 

    	ϕ = √Cϕ * Xmap(trm, randn(eltype_in(trm),size_in(trm)) ./ sqrt.(Ωx(trm)))
		v = ∇!(ϕ[:])   	
    	t = √Ct * Xmap(trm, randn(eltype_in(trm),size_in(trm)) ./ sqrt.(Ωx(trm)))

    	v, t
	end;



	# ------------------------
	L  = FieldLensing.ArrayLense(v, ∇!, 0, 1, 16)
	Lᴴ = L'

	T   = t[:]
	LT  = L  * T
	LᴴT = Lᴴ * T

	L⁻¹LT  = L \ LT  
	L⁻ᴴLᴴT = Lᴴ \ LᴴT

	#-
	T    |> matshow; colorbar();
	LT   |> matshow; colorbar();
	T .- LT   |> matshow; colorbar();


	T    |> matshow; colorbar();
	LᴴT  |> matshow; colorbar();
	T .- LᴴT  |> matshow; colorbar();


	T .- L⁻¹LT	|> matshow; colorbar();
	T .- L⁻ᴴLᴴT	|> matshow; colorbar();

	#-
	@benchmark $L * $T   # 5.4 seconds
	@benchmark $Lᴴ * $T  # 6.0 seconds





	# Test some different ways to compute (∂(x+tv(x))/∂x) \ v
	# --------------------------

	function test1(p1, p2, t, j11, j21, j12, j22, v1, v2)
		@inbounds for i ∈ eachindex(j11)
			y = SMatrix{2,2}(1 + t*j11[i], t*j21[i], t*j12[i], 1 + t*j22[i]) \ SVector(v1[i], v2[i])
			# y = factorize([1+t*j11[i]  t*j12[i] ; t*j21[i]  1+t*j22[i]]) \ SVector(v1[i], v2[i])
			p1[i]  = y[1]
			p2[i]  = y[2]
		end
	end


	function test2(p1, p2, t, j11, j21, j12, j22, v1, v2)
		@avx for i ∈ eachindex(j11)
			m11  = 1 + t * j22[i] 
			m12  =   - t * j12[i] 
			m21  =   - t * j21[i] 
			m22  = 1 + t * j11[i] 
			dt  = m11 * m22 - m12 * m21
			p1[i]  = (m11 * v1[i] + m12 * v2[i]) / dt
			p2[i]  = (m21 * v1[i] + m22 * v2[i]) / dt
		end
	end


	function test3(p1, p2, t, j11, j21, j12, j22, v1, v2)
		@avx for i ∈ eachindex(j11)
			m11  = 1 + t * j22[i] 
			m12  =   - t * j12[i] 
			m21  =   - t * j21[i] 
			m22  = 1 + t * j11[i] 
			dt   = m11 * m22 - m12 * m21
			m11  /= dt
			m12  /= dt
			m21  /= dt
			m22  /= dt
			p1[i]  = m11 * v1[i] + m12 * v2[i]
			p2[i]  = m21 * v1[i] + m22 * v2[i]
		end
	end



	function test4(p1, p2, t, j11, j21, j12, j22, v1, v2)
		@avx for i ∈ eachindex(j11)
			m11  = 1 + t * j22[i] 
			m12  =   - t * j12[i] 
			m21  =   - t * j21[i] 
			m22  = 1 + t * j11[i]
			dt   = m11 * m22 - m12 * m21 
			dt′  = m11 * m21 + m12 * m22 
			r    = hypot(m12, m22)
			c    = m22 / r
			s    = m12 / r
			p1[i] = (c*v1[i] - s*v2[i]) / (dt/r)
			p2[i] = s*v1[i] + c*v2[i] - (dt′/r) * p1[i]
		end
	end


	t₀ = 0.9f0
	j11, j21, j12, j22 = 𝕁!(v)
	v1, v2 = v 
	p1, p2 = similar(v1), similar(v1)

	@benchmark test1($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
	@benchmark test2($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
	@benchmark test3($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)
	@benchmark test4($p1, $p2, $t₀, $j11, $j21, $j12, $j22, $v1, $v2)


	t₀ = 3.25f0

	p1a, p2a = similar(v1), similar(v1)
	p1b, p2b = similar(v1), similar(v1)
	p1c, p2c = similar(v1), similar(v1)
	p1d, p2d = similar(v1), similar(v1)

	test1(p1a, p2a, t₀, j11, j21, j12, j22, v1, v2)
	test2(p1b, p2b, t₀, j11, j21, j12, j22, v1, v2)
	test3(p1c, p2c, t₀, j11, j21, j12, j22, v1, v2)
	test4(p1d, p2d, t₀, j11, j21, j12, j22, v1, v2)


	matshow(p1a); colorbar()
	matshow(p1b); colorbar()
	matshow(p1c); colorbar()
	matshow(p1d); colorbar()


	matshow(p2a); colorbar()
	matshow(p2b); colorbar()
	matshow(p2c); colorbar()
	matshow(p2d); colorbar()



	matshow(abs.(p1a .- p1b)); colorbar()
	matshow(abs.(p1a .- p1c)); colorbar()
	matshow(abs.(p1a .- p1d)); colorbar()



end
