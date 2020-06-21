# Default for Trn <: 𝕎, m == d
# -------------------------------------------

using FFTransforms: 𝕎

function plan(L::Xlense{d,Trn,Tf,Ti,d}) where {Tf, Ti, d, Trn<:𝕎{Tf,d}}
	szf, szi =  size_in(L.trn), size_out(L.trn)
	k   = FFTransforms.fullfreq(L.trn)
	vx  = tuple((L.v[i][:] for i=1:d)...)
	∂vx = Array{Tf,d}[(DiagOp(Xfourier(L.trn,im*k[c]))*L.v[r])[:] for r=1:d, c=1:d]
	mx  = deepcopy(∂vx)
	px  = deepcopy(vx)
	∇y  = deepcopy(vx)
	sk  = zeros(Ti,szi)
	yk  = zeros(Ti,szi)
	XlensePlan{d,Trn,Tf,Ti,d}(L.trn,k,vx,∂vx,mx,px,∇y,sk,yk)
end

function gradient!(∇y::NTuple{d,A}, y::A, Lp::XlensePlan{d,Trn}) where {Tf, d, A<:Array{Tf,d}, Trn<:𝕎{Tf,d}}
	F = FFTransforms.plan(Lp.trn)
	mul!(Lp.yk, F.unscaled_forward_transform, y)
	for i = 1:d
		@inbounds @. Lp.sk = Lp.yk * Lp.k[i] * im * F.scale_forward * F.scale_inverse
		mul!(∇y[i], F.unscaled_inverse_transform, Lp.sk)
	end
end
