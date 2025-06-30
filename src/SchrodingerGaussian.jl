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
    Discretization parameters
=#

export Discretization
include("Discretization.jl")

#=
    Potential parameters
=#

export Potential
include("Potential.jl")

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
    Solver for Schrodinger equation
=#
export schrodinger_best_gaussian
export schrodinger_gaussian_greedy
export schrodinger_gaussian1d_polynomial_greedy
export schrodinger_gaussian_greedy_timestep
include("schrodinger/schrodinger_solver.jl")
include("schrodinger/schrodinger_greedy.jl")
include("schrodinger/schrodinger_greedy_polynomial.jl")


end
