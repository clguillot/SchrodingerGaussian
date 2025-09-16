using LinearAlgebra
using FFTW

function sine_transform!(u::AbstractVector{T}) where{T<:Number}
    FFTW.r2r!(u, FFTW.RODFT00)
    u ./= (length(u) + 1)
    return u
end

function sine_transform(u::AbstractVector)
    v = copy(u)
    return sine_transform!(v)
end

function sine_inv_transform!(u::AbstractVector)
    FFTW.r2r!(u, FFTW.RODFT00)
    u ./= 2
    return u
end

function sine_inv_transform(u::AbstractVector)
    v = copy(u)
    return sine_inv_transform!(v)
end

function schrodinger_sine(a::T, b::T, Lt::Int, u0, v, M::T, Lx::Int) where{T<:AbstractFloat}

    h = (b - a) / (Lt - 1)
    hx = 2*M / (Lx+1)
    U0 = sine_transform([u0(-M + j*hx) for j=1:Lx])
    v_list = [v(-M + j*hx) for j=1:Lx]

    wave_list = [(π*k/(2*M))^2 for k=1:Lx]

    function Λ!(V, t, U)
        @. V = cis(-t * wave_list) * U
        sine_inv_transform!(V)
        V .*= v_list
        sine_transform!(V)
        @. V = cis(t * wave_list) * V
    end

    #Result
    U = zeros(Complex{T}, Lx, Lt)
    #Buffers
    W = zeros(Complex{T}, Lx)
    μ1 = zeros(Complex{T}, Lx)
    μ2 = zeros(Complex{T}, Lx)
    μ3 = zeros(Complex{T}, Lx)
    μ4 = zeros(Complex{T}, Lx)
    @views U[:, 1] .= U0
    for k=1:Lt-1
        t = a + (k-1)*h
        @views Uk = U[:, k]
        @views Ukp1 = U[:, k+1]

        #μ1
        W .= Uk
        Λ!(μ1, t, W)

        #μ2
        W .= Uk .- (1im*h)/2 .* μ1
        Λ!(μ2, t + h/2, W)

        #μ3
        W .= Uk .- (1im*h)/2 .* μ2
        Λ!(μ3, t + h/2, W)

        #μ4
        W .= Uk .- (1im*h) .* μ3
        Λ!(μ4, t + h, W)
        
        Ukp1 .= Uk .- (1im*h)/6 .* (μ1 .+ 2 .* μ2 .+ 2 .* μ3 .+ μ4)
    end

    for k in 1:Lt
        t = a + (k-1)*h
        @. @views U[:, k] = cis(-t * wave_list) * U[:, k]
    end

    return U
end


function strange_sine_interaction3d(Vfunc, u0func, M, N, dt, steps)

    # Create physical grid in [-M, M]^3, excluding boundaries
    x = range(-M, M, length=N+2)[2:end-1]  # size N
    X = reshape(x, N, 1, 1)     # X(i,j,k) = x[i]
    Y = reshape(x, 1, N, 1)     # Y(i,j,k) = y[j]
    Z = reshape(x, 1, 1, N)     # Z(i,j,k) = z[k]
    u = zeros(ComplexF64, N, N, N, steps+1)

    # Evaluate initial condition and potential
    (@view u[:, :, :, begin]) .= @. ComplexF64(u0func(X, Y, Z))
    V = @. Vfunc(X, Y, Z)

    # Laplacian eigenvalues
    k = collect(1:N)
    λ = @. (k * π / (2M))^2
    Λ = reshape(λ, N, 1, 1) .+ reshape(λ, 1, N, 1) .+ reshape(λ, 1, 1, N)

    #
    eΔ = @. cis(-dt/2 * Λ)
    eV = @. cis(-dt * V)

    norm_factor = (2N+2)^3  # normalization for DST-I (RODFT00)

    plan_tf = FFTW.plan_r2r!(u[:, :, :, begin], FFTW.RODFT00)

    # Applies exp(itΔ) to u in sine basis
    # v operates as a buffer
    function exp_laplacian!(u)
        plan_tf * u
        @. u = eΔ * u / norm_factor
        plan_tf * u
        return u
    end

    # Order 2 splitting
    for n in 1:steps
        t = (n-1) * dt
        u_n = @view u[:, :, :, n+1]
        u_n .= @view u[:, :, :, n]
        exp_laplacian!(u_n)
        @. u_n *= eV
        exp_laplacian!(u_n)
    end

    return u, X, Y, Z
end