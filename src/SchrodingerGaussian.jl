module SchrodingerGaussian

using Base.Threads

using LinearAlgebra
using BlockBandedMatrices
using LineSearches
using ForwardDiff
using DiffResults
using StaticArrays
using HermiteWavePackets



include("utils.jl")
include("block_tridiagonal_system.jl")

export gaussian_approx

include("gaussian_approx.jl/approx_solver.jl")

end
