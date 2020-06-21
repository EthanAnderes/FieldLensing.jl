# FieldLensing


`FieldLensing` is designed to allow the development of stand alone lensing code that operates on arrays, but also to work smoothly with XFields. In what follows we show the basic method for utalizing `Xlense` which is predefined for  ... 



## Lensing on field arrays


```julia
d/dt f = v(t, f)
```

Here is what you need to defined to define for a custom lensing operator on `Array{T,d}` where `T <: Number`. 

First define a struct `MyFlow{Trn,Tf,Ti,d} <: AbstractFlow{Trn,Tf,Ti,d}` where `Trn <: XFields.Transform{Tf,d}`. 
An instance of `MyFlow` must hold sufficient information for specifying a concrete lense operator and also contain fields with names `t₀, t₁, nsteps` representing the start time (of the flow), stop time (of the flow) and `nsteps::Int` which parameterizes the level of discretization of the ODE solver. 

Now 

	* Define `L(v,t,f)` where `v` is a tuple of `Array{Tf,d}`, `t` is a `Real` and `f` is an `Array{Tf,d}`. This computes `v(t, f)` over-writing the storage `v[1], ..., v[d]`.
	* Define `inv(L)` where `L` is a `MyFlow` for the inverse flow (usually by returning another `MyFlow` with time `t₀, t₁` reversed `t₁, t₀`).
	

Now one should be able to do `L * f` and `L \ f` for `L::MyFlow`. 


If the calculation of `v(t, f)` is computationally expensive or memory intensive you may want to pre-process some useful quantities and/or pre-allocated storage before the ODE solver is activated. In which case you can define a low-level intermediate storage type `MyFlowPlan{Trn,Tf,Ti,d}` and additionally do the following.

	* Define `plan(L::MyFlow) -> Lp::MyFlowPlan`.
	* Define `Lp(v,t,f)` for `Lp` a `MyFlowPlan`which computes `v(t, f)` over-writing the storage `v[1], ..., v[d]`.  

In this case, when using an intermediate flow plan, you do not need to define `(L::MyFlow)(...)`. 


## Quickstart




## Notes

Note: I think one can bipass odesolve_RK4 for flow as follows
function FieldLensing.flow(L::Xlense{Trn}, f::Array{Tf,d}) where {Tf,d,Trn<:Transform{Tf,d}}	FieldLensing.flowRK38(L)
end
