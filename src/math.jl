import ForwardDiff
import ForwardDiff: Dual

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.exp(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))

    expv = exp(xx)
    dexpv = expv * dx
    complex(Dual{T,V,N}(real(expv), ForwardDiff.Partials{N,V}(tuple(real(dexpv)...))),
            Dual{T,V,N}(imag(expv), ForwardDiff.Partials{N,V}(tuple(imag(dexpv)...))))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.inv(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))

    invv = inv(xx)
    dinvv = - dx * invv * invv
    complex(Dual{T,V,N}(real(invv), ForwardDiff.Partials{N,V}(tuple(real(dinvv)...))),
            Dual{T,V,N}(imag(invv), ForwardDiff.Partials{N,V}(tuple(imag(dinvv)...))))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.sqrt(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))

    sqrtv = sqrt(xx)
    dsqrtv = dx / (2 * sqrtv)
    complex(Dual{T,V,N}(real(sqrtv), ForwardDiff.Partials{N,V}(tuple(real(dsqrtv)...))),
            Dual{T,V,N}(imag(sqrtv), ForwardDiff.Partials{N,V}(tuple(imag(dsqrtv)...))))
end