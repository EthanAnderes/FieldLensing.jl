

##############################################################################
#=
struct Write_xk_op! and Add_xk_op!
 ....  allows allocation free composition of Fourier and Map diag operators

Instances of Write_xk_op! and Add_xk_op! callable on certain argument
signatures used in the following non-allocating methods

# TODO: I think there is a way to write a generic version fallback

=#
##############################################################################



# struct Add_xk_op!{T<:Real,F}
#     ik::Vector{Matrix{Complex{T}}}
#     s1k::Matrix{Complex{T}}
#     s2k::Matrix{Complex{T}}
#     s3k::Matrix{Complex{T}}
#     s4k::Matrix{Complex{T}}
#     s1x::Matrix{T}
#     s2x::Matrix{T}
#     s3x::Matrix{T}
#     s4x::Matrix{T}
#     FFT::F
# end

# struct Write_xk_op!{T<:Real,F}
#     ik::Vector{Matrix{Complex{T}}}
#     s1k::Matrix{Complex{T}}
#     s2k::Matrix{Complex{T}}
#     s3k::Matrix{Complex{T}}
#     s4k::Matrix{Complex{T}}
#     s1x::Matrix{T}
#     s2x::Matrix{T}
#     s3x::Matrix{T}
#     s4x::Matrix{T}
#     FFT::F
# end

# # constructor
# @generated function Add_xk_op!(::Type{P}, ::Type{T}) where {θ,nside,T<:Real,P<:Flat{θ,nside}}
#     g     = r𝔽(P,T)
#     ik    = Vector{Matrix{Complex{T}}}(undef, 2)
#     ik[1] = complex.(0 .* g.k[2], g.k[1])
#     ik[2] = complex.(0 .* g.k[1], g.k[2])
#     s1k = fill(Complex{T}(0), nside÷2+1, nside) 
#     s2k = fill(Complex{T}(0), nside÷2+1, nside)
#     s3k = fill(Complex{T}(0), nside÷2+1, nside)
#     s4k = fill(Complex{T}(0), nside÷2+1, nside)
#     s1x = fill(T(0), nside, nside)
#     s2x = fill(T(0), nside, nside)
#     s3x = fill(T(0), nside, nside)
#     s4x = fill(T(0), nside, nside)  
#     FFT   =  g.FFT
#     Add_xk_op!{T,typeof(FFT)}(ik, s1k,s2k,s3k,s4k, s1x,s2x,s3x,s4x, FFT)
# end

# # constructor
# @generated function Write_xk_op!(::Type{P}, ::Type{T}) where {θ,nside,T<:Real,P<:Flat{θ,nside}}
#     g     = r𝔽(P,T)
#     ik    = Vector{Matrix{Complex{T}}}(undef, 2)
#     ik[1] = complex.(0 .* g.k[2], g.k[1])
#     ik[2] = complex.(0 .* g.k[1], g.k[2])
#     s1k  = fill(Complex{T}(0), nside÷2+1, nside) 
#     s2k  = fill(Complex{T}(0), nside÷2+1, nside)
#     s3k  = fill(Complex{T}(0), nside÷2+1, nside)
#     s4k  = fill(Complex{T}(0), nside÷2+1, nside)
#     s1x  = fill(T(0), nside, nside)
#     s2x  = fill(T(0), nside, nside)
#     s3x  = fill(T(0), nside, nside)
#     s4x  = fill(T(0), nside, nside) 
#     FFT = g.FFT
#     Write_xk_op!{T,typeof(FFT)}(ik, s1k,s2k,s3k,s4k, s1x,s2x,s3x,s4x, FFT)
# end



struct Add_xk_op!{T<:Real,F,uF}
    ik::Vector{Matrix{Complex{T}}}
    s1k::Matrix{Complex{T}}
    s2k::Matrix{Complex{T}}
    s3k::Matrix{Complex{T}}
    s4k::Matrix{Complex{T}}
    s1x::Matrix{T}
    s2x::Matrix{T}
    s3x::Matrix{T}
    s4x::Matrix{T}
    FFT::F
    uFFT::uF
end

struct Write_xk_op!{T<:Real,F,uF}
    ik::Vector{Matrix{Complex{T}}}
    s1k::Matrix{Complex{T}}
    s2k::Matrix{Complex{T}}
    s3k::Matrix{Complex{T}}
    s4k::Matrix{Complex{T}}
    s1x::Matrix{T}
    s2x::Matrix{T}
    s3x::Matrix{T}
    s4x::Matrix{T}
    FFT::F
    uFFT::uF
end

# constructor
@generated function Add_xk_op!(::Type{P}, ::Type{T}) where {θ,nside,T<:Real,P<:Flat{θ,nside}}
    g     = r𝔽(P,T)
    ik    = Vector{Matrix{Complex{T}}}(undef, 2)
    ik[1] = complex.(0 .* g.k[2], g.k[1])
    ik[2] = complex.(0 .* g.k[1], g.k[2])
    s1k = fill(Complex{T}(0), nside÷2+1, nside) 
    s2k = fill(Complex{T}(0), nside÷2+1, nside)
    s3k = fill(Complex{T}(0), nside÷2+1, nside)
    s4k = fill(Complex{T}(0), nside÷2+1, nside)
    s1x = fill(T(0), nside, nside)
    s2x = fill(T(0), nside, nside)
    s3x = fill(T(0), nside, nside)
    s4x = fill(T(0), nside, nside)  
    FFT   =  g.FFT
    unscaled_FFT = plan_rfft(Matrix{T}(undef,nside,nside); flags=FFTW.ESTIMATE)
    Add_xk_op!{T,typeof(FFT),typeof(unscaled_FFT)}(ik, s1k,s2k,s3k,s4k, s1x,s2x,s3x,s4x, FFT,unscaled_FFT)
end

# constructor
@generated function Write_xk_op!(::Type{P}, ::Type{T}) where {θ,nside,T<:Real,P<:Flat{θ,nside}}
    g     = r𝔽(P,T)
    ik    = Vector{Matrix{Complex{T}}}(undef, 2)
    ik[1] = complex.(0 .* g.k[2], g.k[1])
    ik[2] = complex.(0 .* g.k[1], g.k[2])
    s1k  = fill(Complex{T}(0), nside÷2+1, nside) 
    s2k  = fill(Complex{T}(0), nside÷2+1, nside)
    s3k  = fill(Complex{T}(0), nside÷2+1, nside)
    s4k  = fill(Complex{T}(0), nside÷2+1, nside)
    s1x  = fill(T(0), nside, nside)
    s2x  = fill(T(0), nside, nside)
    s3x  = fill(T(0), nside, nside)
    s4x  = fill(T(0), nside, nside) 
    FFT = g.FFT
    unscaled_FFT = plan_rfft(Matrix{T}(undef,nside,nside); flags=FFTW.ESTIMATE)
    Write_xk_op!{T,typeof(FFT),typeof(unscaled_FFT)}(ik, s1k,s2k,s3k,s4k, s1x,s2x,s3x,s4x, FFT,unscaled_FFT)
end




##############################################################################

# write/add methods (Note: first argument is modified)

##############################################################################


# ----------------------------------------------------------------------------- #
# write/add method:
#   add_op!(x, x, Int, x)
#   write_op!(x, x, Int, x)
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outx::A, px_i::A, i::Int, fx_f::A) where {T,F,uF,A<:Matrix{T}}
    mul!(op.s1k, op.uFFT, fx_f)
    op.s1k .*= op.ik[i]
    ldiv!(op.s1x, op.uFFT, op.s1k)
    outx .+= px_i .* op.s1x
    return nothing
end

@inbounds function (op::Write_xk_op!{T,F,uF})(outx::A, px_i::A, i::Int, fx_f::A) where {T,F,uF,A<:Matrix{T}}
    mul!(op.s1k, op.uFFT, fx_f)
    op.s1k .*= op.ik[i]
    ldiv!(op.s1x, op.uFFT, op.s1k)
    outx .= px_i .* op.s1x
    return nothing
end

# fused version of write_op!
@inbounds function (op::Write_xk_op!{T,F,uF})(outx::A, px_ij::Tuple{A,A}, ij::Tuple{Int,Int}, fx_f::A) where {T,F,uF,A<:Matrix{T}}
    mul!(op.s2k, op.uFFT, fx_f)
    op.s1k  .= op.ik[ij[1]] .* op.s2k
    op.s2k .*= op.ik[ij[2]]
    ldiv!(op.s1x, op.uFFT, op.s1k)
    ldiv!(op.s2x, op.uFFT, op.s2k)
    outx .= px_ij[1] .* op.s1x .+ px_ij[2] .* op.s2x
    return nothing
end


# ----------------------------------------------------------------------------- #
# write/add method: 
#   add_op!(x, Int, x, x)
#   write_op!(x, Int, x, x)
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outx::A, i::Int, px_i::A, fx::A) where {T,F,uF,A<:Matrix{T}}
    op.s1x .= px_i .* fx
    mul!(op.s1k, op.uFFT, op.s1x)
    op.s1k .*= op.ik[i]
    ldiv!(op.s1x, op.uFFT, op.s1k)
    outx .+= op.s1x
    return nothing
end

@inbounds function (op::Write_xk_op!{T,F,uF})(outx::A, i::Int, px_i::A, fx::A) where {T,F,uF,A<:Matrix{T}}
    op.s1x .= px_i .* fx
    mul!(op.s1k, op.uFFT, op.s1x)
    op.s1k .*= op.ik[i]
    ldiv!(outx, op.uFFT, op.s1k)
    return nothing
end

# fused version of write_op!
@inbounds function (op::Write_xk_op!{T,F,uF})(outx::A, ij::Tuple{Int,Int}, px_ij::Tuple{A,A},  fx::A) where {T,F,uF,A<:Matrix{T}}
    op.s1x .= px_ij[1] .* fx
    op.s2x .= px_ij[2] .* fx
    mul!(op.s1k, op.uFFT, op.s1x)
    mul!(op.s2k, op.uFFT, op.s2x)
    op.s1k .= op.ik[ij[1]] .* op.s1k .+ op.ik[ij[2]] .* op.s2k 
    ldiv!(outx, op.uFFT, op.s1k)
    return nothing
end






# ----------------------------------------------------------------------------- #
# write/add method: 
#   add_op!(k, Int, x, x)
#   write_op!(k, Int, x, x) 
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, i::Int, px_i::A, fx::A) where {T,F,uF,A<:Matrix{T},B<:Matrix{Complex{T}}}
    op.s1x .= px_i .* fx
    mul!(op.s1k, op.FFT, op.s1x)
    outk .+= op.s1k .* op.ik[i]
    return nothing
end

@inbounds function (op::Write_xk_op!{T,F,uF})(outk::B, i::Int, px_i::A, fx::A) where {T,F,uF,A<:Matrix{T},B<:Matrix{Complex{T}}}
    op.s1x .= px_i .* fx
    mul!(outk, op.FFT, op.s1x)
    outk .*= op.ik[i]
    return nothing
end


# fused version of write_op!
@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, ij::Tuple{Int,Int}, px_ij::Tuple{A,A},  fx::A) where {T,F,uF,A<:Matrix{T},B<:Matrix{Complex{T}}}
    op.s1x .= px_ij[1] .* fx
    op.s2x .= px_ij[2] .* fx
    mul!(op.s1k, op.FFT, op.s1x)
    mul!(op.s2k, op.FFT, op.s2x)
    outk .+= op.ik[ij[1]] .* op.s1k .+ op.ik[ij[2]] .* op.s2k 
    return nothing
end




# ----------------------------------------------------------------------------- #
# write/add method: 
#   add_op!(k, Real, Int, x, x)
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, t::T, i::Int, px_i::A, δᵀ_fx::A) where {T,F,uF,A<:Matrix{T},B<:Matrix{Complex{T}}}
    op.s1x .= px_i .* δᵀ_fx
    mul!(op.s1k, op.FFT, op.s1x)
    outk .+= t .* op.s1k .* op.ik[i]
    return nothing
end



# ----------------------------------------------------------------------------- #
# add method:
#    add_op!(k, Int, Int, x, x, x, x, Int, x)  
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, p::Int, q::Int, ∂ϕ_j::A, Mt_ip::A, Mt_qj::A, δᵀ_fx::A, i::Int, fx_f::A) where {T,F,uF,A<:Matrix{T}, B<:Matrix{Complex{T}}}
    mul!(op.s1k, op.FFT, fx_f)
    op.s1k .*= op.ik[i]
    ldiv!(op.s1x, op.FFT, op.s1k)
    op.s1x .*= ∂ϕ_j .* Mt_ip .* Mt_qj .* δᵀ_fx
    mul!(op.s1k, op.FFT, op.s1x)
    outk .+= op.ik[p] .* op.ik[q] .* op.s1k
    return nothing
end


# ----------------------------------------------------------------------------- #
# add method: 
#   add_op!(k, Int, x, x, Int, x)  
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, j::Int, Mt_ij::A, δᵀ_fx::A, i::Int, fx_f::A) where {T,F,uF,A<:Matrix{T}, B<:Matrix{Complex{T}}}
    mul!(op.s1k, op.FFT, fx_f)
    op.s1k .*= op.ik[i]
    ldiv!(op.s1x, op.FFT, op.s1k)
    op.s1x .*= Mt_ij .* δᵀ_fx
    mul!(op.s1k, op.FFT, op.s1x)
    outk .+= op.ik[j] .* op.s1k
    return nothing
end


# ----------------------------------------------------------------------------- #
# add method: 
#   add_op!(k, Real, Int, Int, x, x, x, x)
# ----------------------------------------------------------------------------- #

@inbounds function (op::Add_xk_op!{T,F,uF})(outk::B, t::T, p::Int, q::Int, ∂ϕ_j::A, Mt_ip::A, Mt_qj::A, fx_f::A) where {T,F,uF,A<:Matrix{T}, B<:Matrix{Complex{T}}}
    op.s1x .= t .* ∂ϕ_j .* Mt_ip .* Mt_qj .* fx_f
    mul!(op.s1k, op.FFT, op.s1x)
    outk .+= op.ik[p] .* op.ik[q] .* op.s1k
    return nothing
end


# fused version
@inbounds function (op::Add_xk_op!{T,F,uF})(
        outk::B, 
        t::T, 
        p::Tuple{Int,Int,Int,Int}, 
        q::Tuple{Int,Int,Int,Int},  
        ∂ϕ_j::A, 
        Mt_ip::Tuple{A,A,A,A}, 
        Mt_qj::Tuple{A,A,A,A},  
        fx_f::A     ) where {T,F,uF,A<:Matrix{T}, B<:Matrix{Complex{T}}}
    op.s1x .= t .* ∂ϕ_j .* Mt_ip[1] .* Mt_qj[1] .* fx_f
    op.s2x .= t .* ∂ϕ_j .* Mt_ip[2] .* Mt_qj[2] .* fx_f
    op.s3x .= t .* ∂ϕ_j .* Mt_ip[3] .* Mt_qj[3] .* fx_f
    op.s4x .= t .* ∂ϕ_j .* Mt_ip[4] .* Mt_qj[4] .* fx_f
    mul!(op.s1k, op.FFT, op.s1x)
    mul!(op.s2k, op.FFT, op.s2x)
    mul!(op.s3k, op.FFT, op.s3x)
    mul!(op.s4k, op.FFT, op.s4x)
    outk .+= op.ik[p[1]].*op.ik[q[1]].*op.s1k .+  op.ik[p[2]].*op.ik[q[2]].*op.s2k .+  op.ik[p[3]].*op.ik[q[3]].*op.s3k .+ op.ik[p[4]].*op.ik[q[4]].*op.s4k
    return nothing
end

