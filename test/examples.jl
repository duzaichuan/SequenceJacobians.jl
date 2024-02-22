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
calis = [:r => 0.03, :eis => 0.5, :G => 0.2, :B=>0.8, :Y => 1.0]
tars = [:asset_mkt => 0]
inits = [:β => 0.85]
# calis = [:r => 0.03, :eis => 0.5, :G => 0.2, :Y => 1.0]
# tars = [:asset_mkt => 0, :MPC => 0.25]
# inits = [:β => 0.85, :B => 0.8]
ss = SteadyState(m, calis, inits, tars)
solve(Hybrid, ss, ss.inits, ftol=1e-10)
ss[]
D_cum = sum(ss.blks[2].ha.D, dims=2) |> vec |> cumsum
agrid = ss.blks[2].ha.aproc.g
f = Figure()
ax = Axis(f[1,1])
lines!(ax, agrid, D_cum)
xlims!(ax, 0, 5)

T = 300
ρ_g = 0.8
dG = 0.01 .* ρ_g .^ range(0,T-1)
ρ_b = 0.9
dB = cumsum(dG) .* ρ_b .^ range(0,T-1)
J = TotalJacobian(m, [:G, :B, :Y], [:asset_mkt], ss[], T)
gj = GEJacobian(J, [:G, :B])
gs = GMaps(gj)
irfs = impulse(gs, [:G=>dG, :B=>dB])
dYg = irfs[:G][:Y]
dYb = irfs[:G][:Y] .+ irfs[:B][:Y]
dDb = irfs[:G][:deficit] .+ irfs[:B][:deficit]
lines(dYg[1:50])
lines!(dYb[1:50])
lines(dDb[1:50])
dCg = irfs[:G][:C]
dCb = irfs[:G][:C] .+ irfs[:B][:C]
lines(dCg[1:50])
lines!(dCb[1:50])

dgoods = irfs[:G][:goods_mkt]
vec(dY)[1:10]
f = Figure()
ax1 = Axis(f[1,1])
lines!(ax1, 1:50, dG[1:50])
ax2 = Axis(f[1,2])
lines!(ax2, 1:50, dY[1:50])
ax3 = Axis(f[1,3])
lines!(ax3, 1:50, dgoods[1:50])


@simple function nkpc(π, Y, C, θ_w, vphi, frisch, markup_ss, eis, β)
    κ_w = (1- θ_w) * (1 - β * θ_w)/θ_w
    piwres = κ_w * (vphi * Y^(1/frisch) - 1/markup_ss * C^(-1/eis)) + β * lead(π) - π
    return piwres
end

@simple function monetary_taylor(π, ishock, rss, φ_π)
    i = rss + φ_π * π + ishock
    r_ante = i - lead(π)
    return r_ante
end

@simple function ex_post_rate(r_ante)
    r = lag(r_ante)
    return r
end

function hsblocks_add(; hhkwargs...)
    bhh, bfiscal, bmkt = hsblocks()
    bnkpc = nkpc_blk()
    btaylor = monetary_taylor_blk()
    bpr = ex_post_rate_blk()
    return bhh, bfiscal, bmkt, bnkpc, btaylor, bpr
end

m_taylor = model(hsblocks_add())
calis = [:eis => 0.5, :G => 0.2, :B => 0.8, :Y => 1.0, :π=>0, :θ_w=>0.9, :frisch=>1, :markup_ss=>1, :ishock=>0, :rss=>0.03, :φ_π=>1.5]
tars = [:asset_mkt => 0, :piwres=> 0]
inits = [:β => 0.85, :vphi=>1.0]
ss_taylor = SteadyState(m_taylor, calis, inits, tars)
solve(Hybrid, ss_taylor, ss_taylor.inits, ftol=1e-10)
ss_taylor[]

T = 300
ρ_g = 0.8
dG = 0.01 .* ρ_g .^ range(0,T-1)
ρ_b = 0.9
dB = cumsum(dG) .* ρ_b .^ range(0,T-1)
J_taylor = TotalJacobian(m_taylor, [:G, :B, :π, :Y], [:asset_mkt, :piwres], ss_taylor[], T)
gj = GEJacobian(J_taylor, [:G, :B])
gs = GMaps(gj)
irfs_taylor = impulse(gs, [:G=>dG, :B=>dB])

dYb_taylor = irfs_taylor[:G][:Y] .+ irfs_taylor[:B][:Y]
dDb_taylor = irfs_taylor[:G][:deficit] .+ irfs_taylor[:B][:deficit]
series([dYb[1:50] dYb_taylor[1:50]]')
series([dDb[1:50] dDb_taylor[1:50]]')

# @implicit function hh_ra(C=1, A=1, Z=0.8, eis=0.5, r=0.03, β=0.97)
#     euler = (β * (1 + lead(r)))^(-eis) * lead(C) - C
#     budget_constraint = (1 + r) * lag(A) + Z - C - A
#     return (C, A), (euler, budget_constraint), Hybrid
# end

@simple function hh_ra(C, A, Z, eis, r, β)
    euler = (β * (1 + lead(r)))^(-eis) * lead(C) - C
    budget_constraint = (1 + r) * lag(A) + Z - C - A
    return euler, budget_constraint
end

function repre()
    _, bfiscal, bmkt = hsblocks()
    bhh_ra = hh_ra_blk()
    return bhh_ra, bfiscal, bmkt
end

ra = model(repre())
calis = [:r => 0.03, :eis => 0.5, :G => 0.2, :B=>0.8, :Y => 1.0, :β=> (1/(1+0.03))]
tars = [:budget_constraint=>0, :asset_mkt => 0]
inits = [:C => 1, :A=>0.8]
ss_ra = SteadyState(ra, calis, inits, tars)
solve(Hybrid, ss_ra, ss_ra.inits, ftol=1e-10)
T = 300
ρ_g = 0.8
dG = 0.01 .* ρ_g .^ range(0,T-1)
ρ_b = 0.9
dB = cumsum(dG) .* ρ_b .^ range(0,T-1)
J_ra = TotalJacobian(ra, [:G, :B, :A, :C, :Y], [:asset_mkt, :euler, :budget_constraint], ss_ra[], T)
gj = GEJacobian(J_ra, [:G, :B])
gs = GMaps(gj)
irfs_ra = impulse(gs, [:G=>dG, :B=>dB])
dY_ra = irfs_ra[:G][:Y] .+ irfs_ra[:B][:Y]
dD_ra = irfs_ra[:G][:deficit] .+ irfs_ra[:B][:deficit]
dC_ra = irfs_ra[:G][:C] .+ irfs_ra[:B][:C]

@simple function hh_ta(C_RA, A, Z, eis, β, r, λ)
    euler = (β * (1 + lead(r)))^(-eis) * lead(C_RA) - C_RA
    C_H2M = copy(Z)
    C = (1-λ) * C_RA + λ * C_H2M
    budget_constraint = (1 + r) * lag(A) + Z - C - A
    return C, euler, budget_constraint
end

function two_a()
    _, bfiscal, bmkt = hsblocks()
    bhh_ta = hh_ta_blk()
    return bhh_ta, bfiscal, bmkt
end

ta = model(two_a())
calis = [:r => 0.03, :eis => 0.5, :G => 0.2, :B=>0.8, :Y => 1.0, :β=> (1/(1+0.03)), :λ=>0.25]
tars = [:budget_constraint=>0, :asset_mkt => 0]
inits = [:C_RA => 1, :A=>0.8]
ss_ta = SteadyState(ta, calis, inits, tars)
solve(Hybrid, ss_ta, ss_ta.inits, ftol=1e-10)
J_ta = TotalJacobian(ta, [:G, :B, :A, :C_RA, :Y], [:asset_mkt, :euler, :budget_constraint], ss_ta[], T)
gj = GEJacobian(J_ta, [:G, :B])
gs = GMaps(gj)
irfs_ta = impulse(gs, [:G=>dG, :B=>dB])
dY_ta = irfs_ta[:G][:Y] .+ irfs_ta[:B][:Y]
dD_ta = irfs_ta[:G][:deficit] .+ irfs_ta[:B][:deficit]
dC_ta = irfs_ta[:G][:C] .+ irfs_ta[:B][:C]

fig = Figure()
ax1 = Axis(fig[1,1], title="deficit")
series!(ax1, [dDb[1:50] dD_ra[1:50] dD_ta[1:50]]')
ax2 = Axis(fig[1,2], title="Y")
series!(ax2, [dYb[1:50] dY_ra[1:50] dY_ta[1:50]]')
ax3 = Axis(fig[1,3], title="C")
series!(ax3, [dCb[1:50] dC_ra[1:50] dC_ta[1:50]]', labels=["HA", "RA", "TA"])
axislegend()
