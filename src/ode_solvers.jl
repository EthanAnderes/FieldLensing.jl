# The nice format here from Marius in CMBLensing.jl

# RK 4 rule
function odesolve_RK4(f!, y₀, t₀, t₁, nsteps)
    h, h½, h⅙ = (t₁-t₀)/nsteps ./ (1,2,6)
    y, y′  = deepcopy(y₀), similar(y₀)
    v₁, v₂ = similar(y₀), similar(y₀)
    v₃, v₄ = similar(y₀), similar(y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        f!(v₁, t, y)
        f!(v₂, t + h½, (@. y′ = y + h½*v₁))
        f!(v₃, t + h½, (@. y′ = y + h½*v₂))
        f!(v₄, t + h,  (@. y′ = y + h*v₃))
        @. y += h*(v₁ + 2v₂ + 2v₃ + v₄)/6
    end
    return y
end

# RK 3/8 rule
function odesolve_RK38(f!, y₀, t₀, t₁, nsteps)
    ϵ = (t₁-t₀)/nsteps
    h, h⅓, h⅔ = ϵ .* (1, 1//3, 2//3) # time increments
    y, y′  = deepcopy(y₀), similar(y₀)
    v₁, v₂ = similar(y₀), similar(y₀)
    v₃, v₄ = similar(y₀), similar(y₀)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        f!(v₁, t, y)
        f!(v₂, t + h⅓, (@. y′ = y + h⅓*v₁))
        f!(v₃, t + h⅔, (@. y′ = y - h⅓*v₁ + h*v₂))
        f!(v₄, t + h,  (@. y′ = y + h*v₁  - h*v₂ + h*v₃))
        @. y += h*(v₁ + 3v₂ + 3v₃ + v₄)/8
    end
    return y
end


