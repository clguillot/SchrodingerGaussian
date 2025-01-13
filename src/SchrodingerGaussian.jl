module SchrodingerGaussian

using Base.Threads

using LinearAlgebra
using StaticArrays
using BlockArrays
using BlockBandedMatrices
using Unroll

using LineSearches
using ForwardDiff
using DiffResults

using HermiteWavePackets

import InteractiveUtils: @code_warntype

#=
    Basics
=#
include("utils.jl")
include("math.jl")
include("block_tridiagonal_system.jl")

#=
    Approximation
=#
export gaussian_approx
include("gaussian_approx.jl/approx_solver.jl")

#=
    Solver for Schrodinger equation
=#
export schrodinger_best_gaussian
export schrodinger_gaussian_greedy
include("schrodinger/schrodinger_solver.jl")
include("schrodinger/schrodinger_greedy.jl")

end
