using TestEnv
using SequenceJacobians
TestEnv.activate()

using AdvancedMH
using Base: has_offset_axes
using BlockArrays
using CSV
using CodecZlib: GzipDecompressorStream
using DataFrames: DataFrame, ncol, nrow
using Distributions
using DynamicHMC
using JSON3
using LinearAlgebra
using LogDensityProblems: logdensity, logdensity_and_gradient
using LoopVectorization
using MCMCChains
using NLopt
using NLsolve
using NonlinearSystems
using Random
using Roots: Brent, Secant
using SequenceJacobians: ArrayToArgs, ar1impulse!, _solve!, _reshape
using SparseArrays
using StructArrays
using TransformVariables: as, asℝ₊, as𝕀

import SequenceJacobians: backwardsolver, forwardsolver

exampledata(name::Union{Symbol,String}) =
    CSV.read(pkgdir(SequenceJacobians)*"/data/$name.csv.gz", DataFrame)

function loadjson(name::Union{Symbol,String})
    stream = open(pkgdir(SequenceJacobians)*"/data/$name.json.gz") |> GzipDecompressorStream
    return copy(JSON3.read(read(stream, String)))
end

struct ImpulseResidual{U<:ImpulseUpdate, V<:AbstractVector}
    u::U
    tars::V
end

function (r::ImpulseResidual)(resids::AbstractVector, θ)
    r.u(θ)
    resids .= _reshape(r.u.vals, length(r.u.vals)) .- r.tars
    return resids
end

using CairoMakie

using SequenceJacobians.RBC
m = model(rbcblocks())
calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
tars = [:goods_mkt=>0, :r=>0.01, :euler=>0, :Y=>1]
inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
ss = SteadyState(m, calis, inits, tars)
solve(Hybrid, ss, ss.inits, ftol=1e-10)
J = TotalJacobian(m, [:Z,:K,:L], [:euler, :goods_mkt], ss[], 300, excluded=(:walras,))


using SequenceJacobians.KrusellSmith
m = model(ksblocks())
calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
inits = [:β=>0.98, :Z=>0.85, :K=>3]
ss =  SteadyState(m, calis, inits, tars)
solve(Hybrid, ss, ss.inits, ftol=1e-10)
series(ss.blks[2].ha.aproc.g, ss.blks[2].ha.c', labels=["skill $i" for i in 1:7])
series(m.pool[1].ha.aproc.g, m.pool[1].ha.c', labels=["skill $i" for i in 1:7])

T = 300
J_firm = TotalJacobian(m, [:Z,:K], [:r, :w], ss[], 5, excluded=(:goods_mkt, :asset_mkt))
J_firm[:Z][:Y](3)
J_firm[:K][:w](3)
J_ha = TotalJacobian(m, [:r,:w], [:C], ss[], 5, excluded=(:goods_mkt, :asset_mkt))
J_ha[:r][:C].maps[1]
J_ha[:r][:C].maps[1] * J_firm[:Z][:r](5)

J = TotalJacobian(m, [:Z,:K], [:asset_mkt], ss[], T, excluded=(:goods_mkt,))
H_z = J[:Z][:asset_mkt]
H_k = J[:K][:asset_mkt]
gj = GEJacobian(J, :Z)
gs = GMaps(gj)
G_r = gs(Matrix{Float64}(undef, T, T), :Z, :r)

ρs = [0.2 0.4 0.6 0.8 0.9]
dZ = 0.01 * ss[:Z] .* ρs .^ range(0,T-1)
dr = G_r * dZ
irfs = 10000 .* dr[1:50, :]
# news shock
dZ = 0.01 .* (range(0,T-1) .== [5 10 15 20 25])
G_k = gs(Matrix{Float64}(undef, T, T), :Z, :K)
dr = G_k * dZ
irfs = dr[1:50, :]
series(irfs')

using SequenceJacobians.HetSim
m = model(hsblocks())
calis = [:r => 0.03, :eis => 0.5, :G => 0.2, :Y => 1.0]
tars = [:asset_mkt => 0, :MPC => 0.25]
inits = [:β => 0.85, :B => 0.8]
ss = SteadyState(m, calis, inits, tars)
solve(Hybrid, ss, ss.inits, ftol=1e-10)
ss[]
