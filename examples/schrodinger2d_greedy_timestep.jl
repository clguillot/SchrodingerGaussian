using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings
using ProgressBars

import LinearAlgebra.BLAS

include("apply_op.jl")

function test_schrodinger_greedy_timestep(a::T, b::T, Lt::Int, nb_steps::Int, nb_terms::Int, newton_nb_iter::Int, ::Type{T}, plot_result) where{T<:AbstractFloat}

    D = 2
    Gtype = GaussianWavePacket{D, Complex{T}, Complex{T}, T, T}

    λ0 = complex(1.0)
    z0 = SVector{D}(fill(complex(0.5), D))
    q0 = SVector{D}([0.0 ; fill(0.0, D-1)])
    p0 = SVector{D}([0.0 ; fill(0.0, D-1)])
    G0 = [GaussianWavePacket(λ0, z0, q0, p0)]


    Gv = Gaussian(1.0, (@SVector fill(1.0, D)), (@SVector zeros(D)))
    v(x) = Gv(x)

    G_list, res = schrodinger_gaussian_greedy_timestep(Gtype, T, a, b, Lt, nb_steps, G0, apply_op, nb_terms; progressbar=true, maxiter=newton_nb_iter, verbose=false)

    println("Residual = ", res)

    if plot_result
        x_list = T.(-10:0.1:10)
        y_list = T.(-10:0.1:10)  # Define the y-dimension
        t_list = zeros(Lt)
        norm_list = zeros(Lt)
    
        g = @gif for k in 1:Lt
            t = a + (k-1) * (b-a) / (Lt-1)
            G = zeros(Gtype, nb_terms)
            
            for j = 1:nb_terms
                tvec = @SVector fill(t, 2)
                G[j] = inv_fourier(unitary_product(2 * tvec, fourier(G_list[j, k])))
            end
            
            t_list[k] = t
            norm_list[k] = norm_L2(WavePacketArray(G))
            
            # Create 2D grid
            δ = 1e-4
            Z = zeros(length(x_list), length(y_list))
            for (i, x) in enumerate(x_list)
                for (j, y) in enumerate(y_list)
                    Z[i, j] = log10(δ + abs2(WavePacketArray(G)(SVector(x, y))))
                end
            end            
            heatmap(x_list, y_list, Z; color=:viridis, clim=(log10(δ), log10(1.4)))
        end fps=30 every (max(round(Int, Lt / (30 * (b-a))), 1))    
        display(g)

        p = plot()
        plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(p, t_list, fill(norm_L2(G0[1]), Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
        display(p)
        # savefig("norm.pdf")
    end
end