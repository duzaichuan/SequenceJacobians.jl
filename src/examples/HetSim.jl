module HetSim

using ..SequenceJacobians
using Distributions

import SequenceJacobians: endoprocs, exogprocs, valuevars, expectedvalues, policies,
    backwardtargets, backward_init!, backward_endo!

export HSHousehold, hshhblock, hsblocks

struct HSHousehold{TF<:AbstractFloat} <: AbstractHetAgent
    aproc::EndoProc{TF,2}
    eproc::ExogProc{TF}
    c::Matrix{TF}
    cnext::Matrix{TF}
    clast::Matrix{TF}
    mpc::Matrix{TF}
    a::Matrix{TF}
    alast::Matrix{TF}
    coh::Matrix{TF}
    cohnext::Matrix{TF}
    Va::Matrix{TF}
    EVa::Matrix{TF}
    D::Matrix{TF}
    Dendo::Matrix{TF}
    Dlast::Matrix{TF}
end

function HSHousehold(amin, amax, Na, ρe, σe, Ne)
    aproc = assetproc(amin, amax, Na, Na, Ne)
    eproc = rouwenhorstexp(ρe, σe, Ne)
    c = Matrix{Float64}(undef, Na, Ne)
    cnext = similar(c)
    clast = similar(c)
    mpc = similar(c)
    a = similar(c)
    alast = similar(c)
    coh = similar(c)
    cohnext = similar(c)
    Va = similar(c)
    EVa = similar(c)
    D = similar(c)
    Dtemp = similar(c)
    Dlast = similar(c)
    return HSHousehold{eltype(c)}(aproc, eproc, c, cnext, clast, mpc, a, alast,
	coh, cohnext, Va, EVa, D, Dtemp, Dlast)
end

endoprocs(h::HSHousehold) = (h.aproc,)
exogprocs(h::HSHousehold) = (h.eproc,)
valuevars(h::HSHousehold) = (h.Va,)
expectedvalues(h::HSHousehold) = (h.EVa,)
policies(h::HSHousehold) = (h.a, h.c, h.mpc)
backwardtargets(h::HSHousehold) = (h.a=>h.alast, h.c=>h.clast)

function backward_init!(h::HSHousehold, r, eis, Z)
    h.coh .= (1 + r) .* grid(h.aproc) .+ Z .* grid(h.eproc)'
    h.Va .= (1 + r) .* (0.1 .* h.coh).^(-1/eis)
end

function backward_endo!(h::HSHousehold, EVa, r, β, eis, Z)
    agrid = grid(h.aproc)
    egrid = grid(h.eproc)
    h.cnext .= (β.*EVa).^(-eis)
    h.coh .= (1 + r) .* agrid .+ Z .* egrid'
    h.cohnext .= h.cnext .+ agrid
    for i in 1:length(egrid)
	interpolate_y!(view(h.a,:,i), view(h.coh,:,i), agrid, view(h.cohnext,:,i))
    end
    # Ensure that asset is always nonnegative
    setmin!(h.a, agrid[1])
    h.c .= h.coh .- h.a
    h.Va .= (1 + r).*h.c.^(-1/eis)
    # Approximate mpc out of wealth, with symmetric differences where possible,
    # exactly setting mpc=1 for constrained agents.
    post_return = (1 + r) .* agrid
    h.mpc[2:end-1, :] .= (h.c[3:end, :] .- h.c[1:end-2, :]) ./ (post_return[3:end] .- post_return[1:end-2])
    h.mpc[1, :] .= (h.c[2, :] .- h.c[1, :]) ./ (post_return[2] - post_return[1])
    h.mpc[end, :] .= (h.c[end, :] .- h.c[end-1, :]) ./ (post_return[end] - post_return[end-1])
    h.mpc[h.a .== agrid[1]] .= 1.0
    h.mpc .= h.mpc .* egrid'
end

@simple function fiscal(B, r, G, Y)
    T = (1 + r) * lag(B) + G - B  # total tax burden
    Z = Y - T  # after tax income
    deficit = G - T
    return T, Z, deficit
end

@simple function mkt_clearing(A, B, Y, C, G)
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt
end

function hshhblock(amin, amax, Na, ρe, σe, Ne; kwargs...)
    kshh = HSHousehold(amin, amax, Na, ρe, σe, Ne)
    return block(kshh, [:r, :β, :eis, :Z], [:A, :C, :MPC]; kwargs...)
end

function hsblocks(; hhkwargs...)
    bhh = hshhblock(0, 1000, 200, 0.9, 0.92, 10; hhkwargs...)
    bfiscal = fiscal_blk()
    # bmkt = block(mkt_clearing, [:A, :B, :Y, :C, :G], [:asset_mkt, :goods_mkt])
    bmkt = mkt_clearing_blk()
    return bhh, bfiscal, bmkt
end

end
