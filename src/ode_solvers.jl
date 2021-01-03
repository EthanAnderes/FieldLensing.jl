# The nice format here from Marius in CMBLensing.jl

# RK 4 rule
function odesolve_RK4(f!, y₀::AbstractArray, t₀, t₁, nsteps)
    h, h½, h⅙ = (t₁-t₀)/nsteps ./ (1,2,6)
    y, y′  = deepcopy(y₀), similar(y₀)
    v₁, v₂ = similar(y₀), similar(y₀)
    v₃, v₄ = similar(y₀), similar(y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        f!(v₁, t, y)
        f!(v₂, t + h½, (@avx @. y′ = y + h½*v₁))
        f!(v₃, t + h½, (@avx @. y′ = y + h½*v₂))
        f!(v₄, t + h,  (@avx @. y′ = y + h*v₃))
        @avx @. y += h*(v₁ + 2v₂ + 2v₃ + v₄)/6
    end
    return y
end

# RK 4 rule on tuples of fields
function odesolve_RK4(f!, y₀::NTuple{m}, t₀, t₁, nsteps) where {m}
    h, h½, h⅙ = (t₁-t₀)/nsteps ./ (1,2,6)
    y, y′  = deepcopy(y₀), map(similar,y₀)
    v₁, v₂ = map(similar,y₀), map(similar,y₀)
    v₃, v₄ = map(similar,y₀), map(similar,y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]        
        for i=1:m
            f!(v₁[i], t, y[i])
            f!(v₂[i], t + h½, (@inbounds @. y′[i] = y[i] + h½*v₁[i]))
            f!(v₃[i], t + h½, (@inbounds @. y′[i] = y[i] + h½*v₂[i]))
            f!(v₄[i], t + h,  (@inbounds @. y′[i] = y[i] + h*v₃[i]))
            @inbounds @. y[i] += h*(v₁[i] + 2v₂[i] + 2v₃[i] + v₄[i])/6
        end
    end
    return y
end

# RK 3/8 rule
function odesolve_RK38(f!, y₀::AbstractArray, t₀, t₁, nsteps)
    ϵ = (t₁-t₀)/nsteps
    h, h⅓, h⅔ = ϵ .* (1, 1//3, 2//3) # time increments
    y, y′  = deepcopy(y₀), similar(y₀)
    v₁, v₂ = similar(y₀), similar(y₀)
    v₃, v₄ = similar(y₀), similar(y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        f!(v₁, t, y)
        f!(v₂, t + h⅓, (@avx @. y′ = y + h⅓*v₁))
        f!(v₃, t + h⅔, (@avx @. y′ = y - h⅓*v₁ + h*v₂))
        f!(v₄, t + h,  (@avx @. y′ = y + h*v₁  - h*v₂ + h*v₃))
        @avx @. y += h*(v₁ + 3v₂ + 3v₃ + v₄)/8
    end
    return y
end

# RK 3/8 rule on tuples
function odesolve_RK38(f!, y₀::NTuple{m}, t₀, t₁, nsteps) where {m}
    ϵ = (t₁-t₀)/nsteps
    h, h⅓, h⅔ = ϵ .* (1, 1//3, 2//3) # time increments
    y, y′  = deepcopy(y₀), map(similar,y₀)
    v₁, v₂ = map(similar,y₀), map(similar,y₀)
    v₃, v₄ = map(similar,y₀), map(similar,y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        f!(v₁, t, y)

        for i=1:m; (@avx @. y′[i] = y[i] + h⅓*v₁[i]); end
        f!(v₂, t + h⅓, y′)

        for i=1:m; (@avx @. y′[i] = y[i] - h⅓*v₁[i] + h*v₂[i]); end
        f!(v₃, t + h⅔,  y′)

        for i=1:m; (@avx @. y′[i] = y[i] + h*v₁[i]  - h*v₂[i] + h*v₃[i]); end
        f!(v₄, t + h,  y′)

        for i=1:m; (@avx @. y[i] += h*(v₁[i] + 3v₂[i] + 3v₃[i] + v₄[i])/8); end
    end
    return y
end

