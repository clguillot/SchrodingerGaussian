struct Discretization
  t0::Float64
  tf::Float64
  Nt::Int64
  G0::AbstractWavePacket
  nb_g #number of Gaussians
  nb_newton #number of Newton steps
  greedy_orthogonal

  function Discretization(t0::T,tf::T,Nt::Int64,G0::AbstractWavePacket,nb_g::Int64,nb_newton::Int64,greedy_orthogonal::Bool) where {T<:AbstractFloat}
    new(t0,tf,Nt,G0,nb_g,nb_newton,greedy_orthogonal)
  end
end