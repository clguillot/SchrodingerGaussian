struct Potential{D}
  V::AbstractWavePacket{D}

  function Potential(V::AbstractWavePacket{D}) where {D}
    new{D}(V)
  end
end

function (V::Potential)(x)
  V.V(x)
end

function Potential(λs, zs, qs)
  G = Gaussian(zero(λs[1]),zero(zs[1]),zero(qs[1]))
  for i in eachindex(λs)
    G += Gaussian(λs[i],zs[i],qs[i])
  end
  Potential(G)
end