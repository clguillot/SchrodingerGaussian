#=
    Computes ∂ₓ₁∂ₓ₂E(X1, X2)
=#
function gaussian_approx_metric_topright_hessian(::Type{Gtype}, X1::AbstractVector{T}, X2::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    X1 = SVector{psize}(X1)
    X2 = SVector{psize}(X2)

    function f(Y1, Y2)
        G1 = unpack_gaussian_parameters(Gtype, Y1)
        G2 = unpack_gaussian_parameters(Gtype, Y2)
        return real(dot_L2(G1, G2))
    end

    ∇₁f(Y1, Y2) = ForwardDiff.gradient(Z -> f(Z, Y2), Y1)
    return ForwardDiff.jacobian(Z -> ∇₁f(X1, Z), X2)
end