using PyCall

"""
implementation of kmeans algorithm with random inital centroids
"""
function kmeans(points,k,n,goal,output)
    start_indices= Vector{Int}(undef,k)
    centroids = Matrix{Float64}(undef,k,k)
    initial_centroids(start_indices,centroids,points,n,k)

    # fininding the best centroid for evey point until converges
    iter = 0;
    while true
        iter += 1
        centroids_sum= zeros(Float64,k,k)
        centroids_count= zeros(Int,k)
        @simd for i=1:n
            index = 1;
            value = sum(((points[i,:]-centroids[1,:]).^2))
            @simd for j=2:k
                tmp = sum(((points[i,:]-centroids[j,:]).^2))
                if(value>tmp)
                    value=tmp
                    index = j
                end
            end
            centroids_sum[index,:] += points[i,:]
            centroids_count[index] += 1
        end
        @simd for j=1:k
            if centroids_count[j] != 0
                centroids_sum[j,:]=centroids_sum[j,:]/centroids_count[j]
            end
        end
        max_change = max(sqrt.(sum((centroids-centroids_sum).^2,dims=2))...)
        centroids = centroids_sum
        (max_change<=0 || iter == 300) && break
    end
    res = Matrix(undef,k+1,k)
    res[1,:]=start_indices
    res[2:end,:]=centroids
    print_output(goal,"spk",output,res) && return
end


"""
function which gets the data points, k (number of required centroids), n (points' dimension)
the function builds the centroids' matrix, and randomly choose K data points
which the initial centroids will be equal to them
"""
function initial_centroids(start_indices,centroids,points,n,k)
    i=1
    np = pyimport("numpy")
    np.random.seed(0)
    rnd = convert(Int,np.random.choice(0:n-1))
    start_indices[1]=rnd
    
    centroids[1,:]=points[rnd+1,:]
    
    while(i!=k)
        D= Vector{Float64}(undef,n)
        D .= Inf
        P = Vector{Float64}(undef,n)
        @simd for l=1:n
            @simd for j=1:i
                D[l]=min(sum((points[l,:]-centroids[j,:]).^2),D[l])
            end
        end
        D_sum= sum(D)
        P = D/D_sum
        i+=1
        rnd = convert(Int,np.random.choice(0:n-1,p=P))
        start_indices[i]=rnd
        centroids[i,:]=points[rnd+1,:]
    end
end