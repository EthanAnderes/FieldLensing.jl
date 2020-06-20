
# Extendable methods: plan, gradient!
# =====================================================

# Default for Trn <: 𝕎
# -------------------------------------------

function plan(L::Xlense{Trn,Tf,Ti,d}) where {Tf,Ti,d,Trn<:FFTransforms.𝕎{Tf,d}}
	szf, szi =  size_in(L.trn), size_out(L.trn)
	k   = FFTransforms.fullfreq(L.trn)
	vx  = tuple((L.v[i][:] for i=1:d)...)
	∂vx = Array{Tf,d}[(DiagOp(Xfourier(L.trn,im*k[c]))*L.v[r])[:] for r=1:d, c=1:d]
	mx  = deepcopy(∂vx)
	px  = deepcopy(vx)
	∇y  = deepcopy(vx)
	sk  = zeros(Ti,szi)
	yk  = zeros(Ti,szi)
	XlensePlan{Trn,Tf,Ti,d}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
end

function gradient!(∇y::NTuple{d,Array{Tf,d}}, y::Array{Tf,d}, Lp::XlensePlan{Trn}) where {Tf,d,Trn<:FFTransforms.𝕎{Tf,d}}
	FFT = FFTransforms.plan(Lp.trn)
	mul!(Lp.yk, FFT.unscaled_forward_transform, y)
	for i = 1:d
		@inbounds @. Lp.sk = Lp.yk * Lp.k[i] * im * FFT.scale_forward * FFT.scale_inverse
		mul!(∇y[i], FFT.unscaled_inverse_transform, Lp.sk)
	end
end

# Default for Trn <: 𝕊
# -------------------------------------------

# Default for Trn <: ℍ
# -------------------------------------------
