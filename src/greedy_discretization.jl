
struct GreedyDiscretization{T<:AbstractFloat}
    t0::T #initial time
    tf::T #final time
    Lt::Int #number of time steps
    nb_terms::Int #number of terms in the greedy algorithm
    nb_iter::Int #number of steps for the gradient descent
    greedy_orthogonal::Bool

    function GreedyDiscretization(t0::T, tf::T, Lt, nb_terms=1, nb_iter=100, greedy_orthogonal=false) where T
        return new{T}(t0, tf, Lt, nb_terms, nb_iter, greedy_orthogonal)
    end
end