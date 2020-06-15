
struct ∂{T<:Real, In<:Integer} 
	∇::Vector{SparseMatrixCSC{T,In}}
end 

struct ∂∂{T<:Real, In<:Integer} 
	∇∇::Matrix{SparseMatrixCSC{T,In}}
end 


#####################################

# constructors

#####################################


function ∂∂{T,In}(∂i::∂{T,In}) where {T<:Real,In<:Integer}
	∇∇ = Matrix{SparseMatrixCSC{T,In}}(undef, 2,2)
	∇∇[1,1], ∇∇[1,2], ∇∇[2,1], ∇∇[2,2]  = copy_union_support(
		∂i.∇[1] * ∂i.∇[1], ∂i.∇[1] * ∂i.∇[2],
		∂i.∇[2] * ∂i.∇[1], ∂i.∇[2] * ∂i.∇[2]
	)
	∂∂{T,In}(∇∇)
end


function ∂∂{T,In}(∂i::∂{T,In}, ϕ∂i::∂{T,In}) where {T<:Real,In<:Integer}
  ∇∇ = Matrix{SparseMatrixCSC{T,In}}(undef, 2,2)
  ∇∇[1,1], ∇∇[1,2], ∇∇[2,1], ∇∇[2,2]  = copy_union_support(
    ∂i.∇[1] * ϕ∂i.∇[1], ∂i.∇[1] * ϕ∂i.∇[2],
    ∂i.∇[2] * ϕ∂i.∇[1], ∂i.∇[2] * ϕ∂i.∇[2]
  )
  ∂∂{T,In}(∇∇)
end

function wrap(dx::T, period::T)::T where T<:Real
	rem(dx + period/2, period, RoundDown) - period/2
end

@inbounds function healpix_∂(nl, m_patch_ind, Icut_col::Array{In,2}, θvec::Vector{T}, φvec::Vector{T}) where {T<:Real, In<:Integer}
    #nztot = sum(Icut_col .> 0)
    #nztot = sum(Icut_col[1:2:end,:] .> 0)
    nztot = sum(Icut_col .> 0) # this is an upper bound
    λ∂x1  = zeros(T,nztot)
    λ∂x2  = zeros(T,nztot)
    rows  = zeros(In,nztot)
    cols  = zeros(In,nztot)
    done_ind = 0
    for l in 1:size(Icut_col,2)
        pix = m_patch_ind[l]
        ict = Icut_col[:, l]
        ict_filter = ict[ict .> 0]
        lnictf     = length(ict_filter)
        #λ∂x1_tmp, λ∂x2_tmp =  bilinear_λ∂x_λ∂y(
        #λ∂x1_tmp, λ∂x2_tmp =  make_λ∂x_λ∂y(
        λ∂x1_tmp, λ∂x2_tmp =  second_λ∂x_λ∂y(
            wrap.(θvec[ict_filter] .- θvec[pix], T(π)), 
            wrap.(φvec[ict_filter] .- φvec[pix], T(2π)), 
            T(0), 
            T(0), 
        )
        curr_rng = (done_ind+1):(done_ind+lnictf)
        λ∂x1[curr_rng] .= λ∂x1_tmp
        λ∂x2[curr_rng] .= λ∂x2_tmp
        rows[curr_rng] .= pix
        cols[curr_rng] .= ict_filter
        done_ind = done_ind + lnictf
    end
    # now take λ∂x1, λ∂x2 and construct the sparse matrix
    ∇θ = sparse(rows[1:done_ind], cols[1:done_ind], λ∂x1[1:done_ind], nl, nl)
    ∇φ = sparse(rows[1:done_ind], cols[1:done_ind], λ∂x2[1:done_ind], nl, nl)
    ∂i = ∂{T,In}(SparseMatrixCSC{T,In}[∇θ, ∇φ])
     # return asym(∂i)
    return merge_support(∂i) # so long as the diagonal is zero we don't need the anti-symmetry
end



function sphere∂(θvec::Vector{T}, φvec::Vector{T}, nn::In) where {T<:Real, In<:Integer}
    nl = length(θvec)
    ∇θ = spzeros(T, In, nl, nl)
    ∇φ = spzeros(T, In, nl, nl)
    ScS = pyimport("scipy.spatial")
    KDT = ScS.cKDTree(hcat(sin.(θvec) .* cos.(φvec), sin.(θvec) .* sin.(φvec), cos.(θvec)))
    for pix in 1:nl
        θpix, φpix = θvec[pix], φvec[pix]
        d_ind, Icut = KDT.query((sin(θpix) * cos(φpix), sin(θpix) * sin(φpix), cos(θpix)), k=nn+1)
        Icut .+= 1
        block_ind = Icut[2:end]
        #λ∂x1, λ∂x2 = bilinear_λ∂x_λ∂y(
        #λ∂x1, λ∂x2 = make_λ∂x_λ∂y(
        λ∂x1, λ∂x2 = second_λ∂x_λ∂y(
            wrap.(θvec[block_ind] .- θpix, T(π)), 
            wrap.(φvec[block_ind] .- φpix, T(2π)), 
            T(0), 
            T(0), 
        )
        ∇θ[pix, block_ind] .= λ∂x1 
        ∇φ[pix, block_ind] .= λ∂x2 
    end
    ∂i = FieldFlows.∂{T,In}(SparseMatrixCSC{T,In}[∇θ, ∇φ])
     # return asym(∂i)
    return merge_support(∂i) # so long as the diagonal is zero we don't need the anti-symmetry
end



function ∂{T,In}(x1::Vector{T}, x2::Vector{T}; nn::Int=8, dist_max::T=T(Inf), period::T=T(Inf)) where {T<:Real,In<:Integer}
	nl = length(x1)
	@assert nl == length(x2)
	∇1 = spzeros(T, In, nl, nl)
	∇2 = spzeros(T, In, nl, nl)
	dist_to_pix = fill(T(0), nl)
	for pix in 1:nl
			x1pix, x2pix = x1[pix], x2[pix]
			dist_to_pix .= sqrt.(abs2.(wrap.(x1 .- x1pix,period)) .+ abs2.(wrap.(x2 .- x2pix,period)))
			Icut1 = findall(dist_to_pix .< dist_max)
			perm = sortperm(dist_to_pix[Icut1])
			block_ind = Icut1[perm[2:(nn+1)]]
            #λ∂x1, λ∂x2 = bilinear_λ∂x_λ∂y( 
            # λ∂x1, λ∂x2 = make_λ∂x_λ∂y( 
            λ∂x1, λ∂x2 = second_λ∂x_λ∂y( 
				wrap.(x1[block_ind] .- x1pix, period), 
				wrap.(x2[block_ind] .- x2pix, period), 
				T(0), #x1pix, 
				T(0), #x2pix,
			)
			∇1[pix, block_ind] .= λ∂x1 
			∇2[pix, block_ind] .= λ∂x2 
	end
	∂i = ∂{T,In}(SparseMatrixCSC{T,In}[∇1, ∇2])
     # return asym(∂i)
	return merge_support(∂i) # so long as the diagonal is zero we don't need the anti-symmetry
end


function torus∂(x1::Vector{T}, x2::Vector{T}, nn::In, period::T) where {T<:Real,In<:Integer}
    nl = length(x1)
    @assert nl == length(x2)
    ∇1 = spzeros(T, In, nl, nl)
    ∇2 = spzeros(T, In, nl, nl)
    
    ScS = pyimport("scipy.spatial")
    KDT = ScS.cKDTree(hcat(x1, x2), boxsize=[period, period])
    
    for pix in 1:nl
        x1pix, x2pix = x1[pix], x2[pix]
        d_ind, Icut = KDT.query((x1pix, x2pix), k=nn+1)
        Icut .+= 1
        block_ind = Icut[2:end]
        #λ∂x1, λ∂x2 = bilinear_λ∂x_λ∂y(
        λ∂x1, λ∂x2 = make_λ∂x_λ∂y(
        #λ∂x1, λ∂x2 = second_λ∂x_λ∂y(
            wrap.(x1[block_ind] .- x1pix, period),
            wrap.(x2[block_ind] .- x2pix, period),
            T(0), T(0)
        )
        ∇1[pix, block_ind] .= λ∂x1 
        ∇2[pix, block_ind] .= λ∂x2 
    end

    ∂i = ∂{T,In}(SparseMatrixCSC{T,In}[∇1, ∇2])
    # return asym(∂i)
    return merge_support(∂i)
end


#####################################

# methods

#####################################


function (op::∂{T,In})(f::Array{T}) where {T<:Real,In<:Integer}
	op.∇[1] * f, op.∇[2] * f
end

function ∂!(rtn1::Array{T}, rtn2::Array{T}, op::∂{T,In}, f::Array{T}) where {T<:Real,In<:Integer}
	mul!(rtn1, op.∇[1], f)
	mul!(rtn2, op.∇[2], f)
	rtn1, rtn2 
end

function asym(∂i::∂{T,In}) where {T<:Real,In<:Integer}
	∂iᵀ    = ∂ᵀ(∂i)
	∇ = Vector{SparseMatrixCSC{T,In}}(undef, 2)
	∇[1], ∇[2] = copy_union_support(
		(∂i.∇[1] .- ∂iᵀ.∇[1]) ./ 2, 
		(∂i.∇[2] .- ∂iᵀ.∇[2]) ./ 2,
	)
	return ∂{T,In}(∇)
end

function merge_support(∂i::∂{T,In}) where {T<:Real,In<:Integer}
    ∇ = Vector{SparseMatrixCSC{T,In}}(undef, 2)
    ∇[1], ∇[2] = copy_union_support(∂i.∇[1], ∂i.∇[2])
    return ∂{T,In}(∇)
end


function ∂ᵀ(∂i::∂{T,In}) where {T<:Real,In<:Integer}
	∇ᵀ = Vector{SparseMatrixCSC{T,In}}(undef, 2)
	∇ᵀ[1], ∇ᵀ[2] = copy_union_support(
		∂i.∇[1] |> transpose |> SparseMatrixCSC{T,In}, 
		∂i.∇[2] |> transpose |> SparseMatrixCSC{T,In},
	)
	return ∂{T,In}(∇ᵀ)
end

function ∂ᵀ∂ᵀ(∂i∂j::∂∂{T,In}) where {T<:Real,In<:Integer}
	∇ᵀ∇ᵀ = Matrix{SparseMatrixCSC{T,In}}(undef, 2,2)
	∇ᵀ∇ᵀ[1,1], ∇ᵀ∇ᵀ[1,2], ∇ᵀ∇ᵀ[2,1], ∇ᵀ∇ᵀ[2,2] = copy_union_support(
		∂i∂j.∇∇[1,1] |> transpose |> SparseMatrixCSC{T,In}, 
		∂i∂j.∇∇[1,2] |> transpose |> SparseMatrixCSC{T,In}, 
		∂i∂j.∇∇[2,1] |> transpose |> SparseMatrixCSC{T,In}, 
		∂i∂j.∇∇[2,2] |> transpose |> SparseMatrixCSC{T,In}, 
	)
	return ∂∂{T,In}(∇ᵀ∇ᵀ)
end

function copy_union_support(Ms::Vararg{SM,n}) where {T<:Real, In<:Integer,n,SM<:SparseMatrixCSC{T,In}}
	union_support = sum(abs.(m) for m in Ms)
	rtn_vec = SM[zero(union_support) for i in 1:n]
	for mi in 1:n
		for (i,j,v) in zip(findnz(Ms[mi])...)
			rtn_vec[mi][i,j] = v
		end
	end
	return (m for m in rtn_vec)
end


# local regression to second order polynomial
function second_λ∂x_λ∂y(x::Vector{T}, y::Vector{T}, x0::T, y0::T) where T<:Real
    n    = length(x)
    X = hcat(fill(T(1),n), x, y, x.^2, x.*y, y.^2)
    ∂x0X = hcat(T(0), T(1), T(0), 2 .* x0, y0, T(0))
    ∂y0X = hcat(T(0), T(0), T(1), T(0), x0, 2 .* y0)
    QR = qr(X, Val(true))
    hat_prepiv = QR.R \ transpose(QR.Q[:,1:length(∂y0X)])
    hat = QR.P * hat_prepiv
    vec(∂x0X * hat), vec(∂y0X * hat) 
end


# # local regression to first order polynomial
function make_λ∂x_λ∂y(x::Vector{T}, y::Vector{T}, x0::T, y0::T) where T<:Real
  n    = length(x)
  X    = hcat(fill(T(1),n),x, y)
  ∂x0X = hcat(T(0),T(1), T(0))
  ∂y0X = hcat(T(0),T(0), T(1))
  QR   = qr(X, Val(true))
  hat_prepiv = QR.R \ transpose(QR.Q[:,1:length(∂y0X)])
  hat = QR.P * hat_prepiv
  vec(∂x0X * hat), vec(∂y0X * hat) 
end

# Bilinear interpolation local regression to first order polynomial
function bilinear_λ∂x_λ∂y(x::Vector{T}, y::Vector{T}, x0::T, y0::T) where T<:Real
  n    = length(x)
  X    = hcat(fill(T(1),n), x, y, x.*y)
  ∂x0X = hcat(T(0),T(1), T(0), y0)
  ∂y0X = hcat(T(0),T(0), T(1), x0)
  QR   = qr(X, Val(true))
  hat_prepiv = QR.R \ transpose(QR.Q[:,1:length(∂y0X)])
  hat = QR.P * hat_prepiv
  vec(∂x0X * hat), vec(∂y0X * hat) 
end




