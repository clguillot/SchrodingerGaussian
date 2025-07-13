using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings

using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings

include("reference_solution.jl")

function plot_schrodinger2d_gaussian_timestep(pb::GreedyDiscretization{T}, nb_timesteps, G0::AbstractWavePacket{D}, V::AbstractWavePacket{D}; plot_result=true) where{T, D}

    function propag(t, G::AbstractWavePacket{D}) where D
        return inv_fourier(unitary_product(fourier(G), SVector{D}(ntuple(_ -> 2*t, D)...)))
    end
    function apply_op(t, G)
        return propag(-t, V * propag(t, G))
    end
    G_list, res_list = schrodinger_greedy_gaussian_timestep(pb, nb_timesteps, G0, apply_op; progressbar=true)

    if plot_result
        x_list = T.(-10:0.1:10)
        y_list = T.(-10:0.1:10)  # Define the y-dimension
        t_list = zeros(pb.Lt)
        norm_list = zeros(pb.Lt)
    
        g = @gif for k in 1:pb.Lt
            t = pb.t0 + (k-1) * (pb.tf - pb.t0) / (pb.Lt-1)
            G = WavePacketSum{D}(propag.(t, G_list[:, k]))
            
            t_list[k] = t
            norm_list[k] = norm_L2(G)
            
            # Create 2D grid
            δ = 1e-4
            Z = zeros(length(x_list), length(y_list))
            for (i, x) in enumerate(x_list)
                for (j, y) in enumerate(y_list)
                    Z[i, j] = log10(δ + abs2(G(SVector(x, y))))
                end
            end            
            heatmap(x_list, y_list, Z; color=:viridis, clim=(log10(δ), log10(1.4)))
        end fps=30 every (max(round(Int, pb.Lt / (30 * (pb.tf - pb.t0))), 1))    
        display(g)

        p = plot()
        plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(p, t_list, fill(norm_L2(G0), pb.Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
        display(p)
        # savefig("norm.pdf")
    end

    println("res_list = ", res_list)

    # p = plot()
    # plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
    # plot!(p, t_list, fill(norm_L2(G0), pb.Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
    # display(p)
end

G0 = GaussianWavePacket(complex(1.0), SVector(complex(1.0), complex(1.0)), SVector(6.0, 0.0), SVector(-1.0, 0.0))
G0 = G0 / norm_L2(G0)
pb = GreedyDiscretization(0., 10., 101, 20, 100, true)
V = Gaussian(-5.0, SVector(0.5, 0.5), SVector(0.1, 0.0))
plot_schrodinger2d_gaussian_timestep(pb, 10, G0, V)