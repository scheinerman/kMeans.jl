module kMeans

using LinearAlgebra
export L2, kmeans

"""
    _average(list::Vector{T})::T where T

return the average value of the elements in `list`.
"""
function _average(list::Vector{T}) where {T}
    sum(list) / length(list)
end

"""
    L2(a,b)

Compute the L2 distance between `a` and `b`, i.e., `norm(a-b)`.
"""
function L2(a, b)
    return norm(a - b)
end

"""
    random_split(list::Vector{T}, k::Int)::Dict{T,Int}

Randomly assign the elements of `list` to one of `k` parts.
Return a dictionary `parts` where `parts[x]` is the part containing `x`.
"""
function random_split(list::Vector{T}, k::Int)::Dict{T,Int} where {T}
    if k < 1
        error("Number of parts in `random_split` must be positive")
    end

    parts = Dict{T,Int}()

    for x in list
        parts[x] = mod1(rand(Int), k)
    end
    return parts
end


"""
    _make_anchors(parts::Dict{T,Int})::Dict{Int,T} where {T}

Given a labeling of the data, find the centers of mass for each label. 
"""
function _make_anchors(parts::Dict{T,Int})::Dict{Int,T} where {T}
    anchor = Dict{Int,T}()

    for j in unique(values(parts))
        sub_list = [x for x in keys(parts) if parts[x] == j]  # all elements in part j

        anchor[j] = _average(sub_list)
    end
    return anchor
end


"""
    _find_nearest(x::T, anchor::Dict{Int, T}, dist::Function)::Int

Given a data point `x` determine `j` for which `anchor[j]` is closest to `x`.
"""
function _find_nearest(x::T, anchor::Dict{Int,T}, dist::Function)::Int where {T}
    # find the distance from x to any of the anchors
    best_j = first(keys(anchor))
    a = anchor[best_j]
    best_d = dist(a, x)

    for j in keys(anchor)
        a = anchor[j]
        d = dist(a, x)
        if d < best_d
            best_d = d
            best_j = j
        end
    end

    return best_j
end

"""
    _one_step(parts::Dict{T,Int}, dist::Function)::Dict{T,Int} where T

Perform one step of the `kMeans` algorithm.
"""
function _one_step(parts::Dict{T,Int}, dist::Function)::Dict{T,Int} where {T}
    anchor = _make_anchors(parts)
    new_parts = Dict{T,Int}()
    for x in keys(parts)
        new_parts[x] = _find_nearest(x, anchor, dist)
    end

    return new_parts
end


"""
    kmeans(
        list::Vector{T}, k::Int = 2;
        dist::Function = L2, max_steps::Int = 10, verbose::Bool = true
    )::Dict{T,Int} where {T}

Compute the `k`-means partition of the data in `list`. Default is `k` equal to 2.

Returns a dictionary in which data element `x` is assigned an integer between `1` and `k`.

The distance function `dist` defaults to `L2`. Algorithm stops when there is no 
further changes to the resulting partition, or when `max_steps` has been reached.
"""
function kmeans(
    list::Vector{T},
    k::Int = 2;
    dist::Function = L2,
    max_steps::Int = 10,
    verbose::Bool = true
)::Dict{T,Int} where {T}

    if k < 1
        error("Number of parts must be positive")
    end

    parts = random_split(list, k)
    count = 0
    while true
        count += 1
        if count > max_steps
            if verbose
                println("\nmax steps reached")
            end
            return parts
        end

        if verbose
            print(count, " ")
        end
        new_parts = _one_step(parts, dist)
        if new_parts == parts
            println()
            return parts
        end
        parts = new_parts
    end
end

end # module kMeans
