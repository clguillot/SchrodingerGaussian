module SchrodingerGaussian

using Base.Threads
using InteractiveUtils

using LinearAlgebra
using StaticArrays
using BlockArrays
using BlockBandedMatrices
using Unroll

using LineSearches
using ForwardDiff
using DiffResults

using HermiteWavePackets

using ProgressBars

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
include("approx/approx_solver.jl")

#=
    Greedy problem
=#
export GreedyDiscretization
include("greedy_discretization.jl")

#=
    Solver for Schrodinger equation
=#
export schrodinger_greedy_gaussian
export schrodinger_greedy_gaussian_timestep
export schrodinger_greedy_hermite
export schrodinger_greedy_hermite_timestep
include("schrodinger/schrodinger_solver.jl")
include("schrodinger/schrodinger_greedy_gaussian.jl")
include("schrodinger/schrodinger_greedy_hermite.jl")

end
