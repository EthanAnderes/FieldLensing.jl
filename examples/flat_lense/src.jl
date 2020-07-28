
# Using Xlense from FieldLensing
# ==========

# Load modules
# ----------

using FFTW
FFTW.set_num_threads(5)

using BenchmarkTools
using LBblocks
using Spectra
using XFields
using FieldLensing
using FFTransforms
using PyPlot


# Set fourier grid parameters
# ----------

trn = @sblock let θpix′ = 2.5, nside = 512
    θpix = deg2rad(θpix′/60)
    period = nside * θpix
    𝕨 = r𝕎32(nside, period) ⊗ 𝕎(nside, period)
    ordinary_scale(𝕨)*𝕨
end

# Compute the spectral matrices which mimic CMB tempurature and lesing potential
# ------------------------------

Ct, Cϕ = @sblock let trn
    l   = wavenum(trn)
    #cTl = Spectra.cTl_approx.(l)
    cTl = Spectra.cTl_besselj_approx.(l)
    #cTl = Spectra.cTl_matern_cov_approx.(l)
    cϕl = Spectra.cϕl_approx.(l) 

    Ct  = DiagOp(Xfourier(trn, cTl)) 
    Cϕ  = DiagOp(Xfourier(trn, cϕl)) 

    Ct, Cϕ
end;

# Visualize how close the fluctuations look. 
# ---------------

T, ϕ = @sblock let trn, Ct, Cϕ
    zTx = randn(eltype_in(trn),size_in(trn)) ./ √Ωx(trn)
    zϕx = randn(eltype_in(trn),size_in(trn)) ./ √Ωx(trn)

    T = √Ct * Xmap(trn, zTx)
    ϕ = √Cϕ * Xmap(trn, zϕx)

    ## extra scale factor for larger lense
    sc = 1.5 

    T, sc * ϕ
end;

#- 
T[:] |> matshow; colorbar();

#-
ϕ[:] |> matshow; colorbar();


# Construct lense and adjoint lense
# ---------------

# Start by defining the gradient operator and use it to compute the lensing displacements

∇ = @sblock let trn
    k = fullfreq(trn)
    ik1 = Xfourier(trn, im .* k[1]) |> DiagOp
    ik2 = Xfourier(trn, im .* k[2]) |> DiagOp
    (ik1, ik2)
end;

#- 

vϕ = (∇[1] * ϕ, ∇[2] * ϕ)

# Now construct the lensing and adjoint lensing operator

L, Lʰ = @sblock let trn, vϕ, nsteps=16
    t₀ = 0
    t₁ = 1
    nsteps = 16 
    L  = FieldLensing.Xlense(trn, vϕ, t₀, t₁, nsteps)
    L, L' 
end;

# Lense the field
# ---------------

# Forward lensing field

lenT1 = L * T
lenT1[:] |> matshow; colorbar();

# Difference between lensed and un-lensed

(T - lenT1)[:] |> matshow; colorbar();

# Invert the lense and compare

T1 = L \ lenT1
(T - T1)[:] |> matshow; colorbar();


# adjoint Lense the field
# ---------------

# Forward adjoint lensing field

lenʰT1 = Lʰ * T

# Difference between adjoint lensing and un-lensed

(T - lenʰT1)[:] |> matshow; colorbar();

# Invert the lense and compare

ʰT1 = Lʰ \ lenʰT1
(T - T1)[:] |> matshow; colorbar();


# The lense objects are AbstractLinearOps
# ---------------
# ... so they work well with DiagOps 

A = DiagOp(Xmap(ϕ))
B = A * L / Ct * L' * A';

#-
f1 = B * T
f2 = A * (L * (Ct \ (L' * (A' * T))))
√sum(abs2, (f1 - f2)[:])


# Finally some benchmarks
# ----------------

@benchmark $L * $T

#-

@benchmark $Lʰ * $T

