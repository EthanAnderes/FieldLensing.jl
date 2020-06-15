
# -------- field2vec
function field2vec(f::Field{P,T}) where {T<:Real, P<:Pix}
    fx_tuple = f |> LenMapBasis |> data
    return Vector{T}[vec(fx) for fx in fx_tuple] 
end

# ----- deepcopy version of field2vec
f2v(f::Field{P,T,S0}) where {P,T}  = deepcopy(field2vec(f)[1])
f2v(f::Field{P,T,S2}) where {P,T}  = deepcopy(field2vec(f))
f2v(f::Field{P,T,S02}) where {P,T} = deepcopy(field2vec(f))

# -------- vec2field
function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,θ,n,P<:Flat{θ,n},S<:S0} 
    @assert length(f) == 1
    Tmap{P,T}(reshape(f[1],n,n))
end

function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,P<:Healpix,S<:S0} 
    @assert length(f) == 1
    Tsphere{P,T}(f[1])
end

function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,θ,n,P<:Flat{θ,n},S<:S2} 
    @assert length(f) == 2
    QUmap{P,T}(reshape(f[1],n,n), reshape(f[2],n,n))
end

function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,P<:Healpix,S<:S2} 
    @assert length(f) == 2
    QUsphere{P,T}(f[1], f[2])
end

function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,θ,n,P<:Flat{θ,n},S<:S02} 
    @assert length(f) == 3
    TQUmap{P,T}(reshape(f[1],n,n), reshape(f[2],n,n), reshape(f[3],n,n))
end

function vec2field(::Type{P},::Type{T},::Type{S}, f::Vector{Vector{R}}) where {R<:Real,T<:Real,P<:Healpix,S<:S02} 
    @assert length(f) == 3
    TQUsphere{P,T}(f[1], f[2], f[3])
end

# -------- make flow integrate with Fields
function flow(L::AbstractMapFlow{T,In}, f::Field{P,T,S}) where {T<:Real,In<:Integer,P<:Pix,S<:Spin} 
    vec2field(P,T,S, flow(L, field2vec(f)))
end
