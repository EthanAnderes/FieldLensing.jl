# Automatic extension of the basic partial derivaties supplied by the user
# ========================================================

# To use ArrayLense one needs to define a custom 
# struct which computes the numerical partial derivatives

#  ∇!(des, y, Val(1))  ->  ∇₁ y
#   ⋮
#  ∇!(des, y, Val(m))  ->  ∇ₘ y


# Here is an example of how to define a gradient type that can be used 
# With ArrayLense. 
# ----------------------------------- 
# struct Nabla!{Tθ,Tφ} <: Gradient{2}
#     ∂θ::Tθ
#     ∂φᵀ::Tφ
# end
# 
# function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{1}) where {Tθ,Tφ} 
#     mul!(des, ∇!.∂θ, y)
# end
# 
# function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{2}) where {Tθ,Tφ}
#     mul!(des, y, ∇!.∂φᵀ)
# end 
# 

# Note: for this example one can also define adjoint:
# 
# function LinearAlgebra.adjoint(∇!::Nabla!)
#     return Nabla!(
#         ∇!.∂θ',
#         ∇!.∂φᵀ',
#     )
# end


abstract type Gradient{m} end

# apply on tuple arguments with pre-storage

function (∇!::Gradient{m})(∇y::NTuple{m}, y::NTuple{m}) where {m}
    for i=1:m
        ∇!(∇y[i], y[i], Val(i))
    end
    ∇y
end
function (∇!::Gradient{1})(∇y::NTuple{1}, y::NTuple{1})
    ∇!(∇y[1], y[1], Val(1))
    ∇y
end
function (∇!::Gradient{2})(∇y::NTuple{2}, y::NTuple{2})
    ∇!(∇y[1], y[1], Val(1))
    ∇!(∇y[2], y[2], Val(2))
    ∇y
end


# apply on Array arguments with storage

function (∇!::Gradient{m})(∇y::NTuple{m}, y::AbstractArray) where {m}
    for i=1:m
        ∇!(∇y[i], y, Val(i))
    end
end
function (∇!::Gradient{1})(∇y::NTuple{1}, y::AbstractArray)
    ∇!(∇y[1], y, Val(1))
end
function (∇!::Gradient{2})(∇y::NTuple{2}, y::AbstractArray)
    ∇!(∇y[1], y, Val(1))
    ∇!(∇y[2], y, Val(2))
end


# apply on tuple arguments without

function (∇!::Gradient{m})(y::NTuple{m}) where {m}
    ∇y = map(similar, y)
    ∇!(∇y, y)
    ∇y
end


# apply on Array arguments without storage

function (∇!::Gradient{m})(y::AbstractArray) where {m}
    ∇y = tuple((similar(y) for i = Base.OneTo(m))...)
    ∇!(∇y, y)
    ∇y
end




