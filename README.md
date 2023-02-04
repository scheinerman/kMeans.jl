# kMeans


Given a list of data points, use `kmeans` to partition into clusters 
using the `k`-means algorithm.

`kmeans(data, k; dist, max_steps, verbose)` where
* `data` is a list (`Vector`) of data, 
* `k` is an integer giving the desired number of parts (default is `2`), 
* `dist` is a distance function (default is `L2`),
* `max_steps` is the maximum number of steps the algorithm will take before giving up (default is `10`), and
* `verbose` is a boolean value determining if some extra information is printed while the algorithm runs (default is `true`).

The output is a dictionary mapping the elements of `data` to integers in the range from `1` to `k`. 


The default distance function, `L2`, gives the standard Euclidean distance for points
in Euclidean space. 