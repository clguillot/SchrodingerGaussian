import ForwardDiff
import ForwardDiff: Dual

# # Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
@inline function Base.exp(d::Complex{<:Dual{T}}) where T
    FD = ForwardDiff
    x = complex(FD.value(real(d)), FD.value(imag(d)))
    val = exp(x)
    deriv = val
    re_deriv, im_deriv = reim(deriv)
    re_partials = FD.partials(real(d))
    im_partials = FD.partials(imag(d))
    re_retval = FD.dual_definition_retval(Val{T}(), real(val), re_deriv, re_partials, -im_deriv, im_partials)
    im_retval = FD.dual_definition_retval(Val{T}(), imag(val), im_deriv, re_partials, re_deriv, im_partials)
    return complex(re_retval, im_retval)
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
@inline function Base.inv(d::Complex{<:Dual{T}}) where T
    FD = ForwardDiff
    x = complex(FD.value(real(d)), FD.value(imag(d)))
    val = inv(x)
    deriv = - val*val
    re_deriv, im_deriv = reim(deriv)
    re_partials = FD.partials(real(d))
    im_partials = FD.partials(imag(d))
    re_retval = FD.dual_definition_retval(Val{T}(), real(val), re_deriv, re_partials, -im_deriv, im_partials)
    im_retval = FD.dual_definition_retval(Val{T}(), imag(val), im_deriv, re_partials, re_deriv, im_partials)
    return complex(re_retval, im_retval)
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
@inline function Base.sqrt(d::Complex{<:Dual{T}}) where T
    FD = ForwardDiff
    x = complex(FD.value(real(d)), FD.value(imag(d)))
    val = sqrt(x)
    deriv = inv(2*val)
    re_deriv, im_deriv = reim(deriv)
    re_partials = FD.partials(real(d))
    im_partials = FD.partials(imag(d))
    re_retval = FD.dual_definition_retval(Val{T}(), real(val), re_deriv, re_partials, -im_deriv, im_partials)
    im_retval = FD.dual_definition_retval(Val{T}(), imag(val), im_deriv, re_partials, re_deriv, im_partials)
    return complex(re_retval, im_retval)
end

# Multiplication by imaginary unit 1im
@inline function im_unit_mul(z::Complex{T}) where{T}
    x, y = reim(z)
    return complex(-y, x)
end