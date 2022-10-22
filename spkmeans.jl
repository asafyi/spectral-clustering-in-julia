using CSV
using DataFrames
using Tables
using LinearAlgebra
using Printf
using PrettyTables
Base.show(io::IO,f::Float64)= @printf(io,"%.4f",f)
include("kmeans.jl")



function main(k,goal,input,output=nothing)
    # checking input arguments and reading the file
    k = parse(Int,ARGS[1])
    goal = ARGS[2]
    input = ARGS[3]
    output = nothing
    if size(ARGS,1)==4
        output = ARGS[4]
        println(ARGS[4])
    end

    data = CSV.read(input,Tables.matrix, header=0, delim=",")
    n = size(data,1)

    # calculates jacobi on the data matrix
    if(goal=="jacobi")
        jacob = jacobi(data,n)
        print_output(goal,"jacobi",output,jacob,"dataframe") && return
        return
    end

    # else... calculate the weighted matrix
    weight = Matrix{Float64}(undef,n,n)
    for i=1:n, j=1:n
        if i==j
            weight[i,j]=0;
        else
            weight[i,j]=exp(-(norm(data[i,:]-data[j,:])/2))
        end
    end
    print_output(goal,"wam",output,weight) && return

    # calculates the diagonal matrix
    diagonal = zeros(Float64,n,n)
    @simd for i=1:n
        diagonal[i,i]=sum(weight[i,:])
    end
    print_output(goal,"ddg",output,diagonal) && return

    # calculates the lnorm matrix
    diagonal = diagonal^-0.5
    lnorm = I-((diagonal*weight*diagonal))
    print_output(goal,"lnorm",output,lnorm) && return
    
    # using jacobi on lnorm matrix
    jacob = jacobi(lnorm,n)
    print_output(goal,"jacobi2",output,jacob,"dataframe") && return

    # sorting the eigenvactors according to the eigenvalues
    sort!(jacob,alg=InsertionSort,rev=false)
    print_output(goal,"sorted",output,jacob,"dataframe") && return

    # calculates the U matrix
    if k==0
        max = -1
        for i::Int=1:floor(n/2)
            δ= abs(jacob[i,"eigenvalue"]-jacob[i+1,"eigenvalue"])
            if (δ>max)
                k=i
                max=δ
            end
        end
    end

    U = Matrix{Float64}(undef,n,k)
    @simd for i=1:k
        U[:,i]=jacob[i,"eigenvector"]
    end
    print_output(goal,"U",output,U) && return

    # calculates the T normlized matrix
    T = Matrix{Float64}(undef,n,k)
    @simd for i=1:n
        norm_row=norm(U[i,:])
        if (norm_row != 0)
            T[i,:] = (U[i,:]/norm_row)
        else
            T[i,:] = U[i,:]
        end
    end
    print_output(goal,"T",output,T) && return
    kmeans(T,k,n,goal,output)
end




"""
The function gets the goal which was chosen, the goal which fits the current state, the output file path (can be nothing),
the dataframe/table data which should be printed and type of data - table or dataframe. prints to file or console.
return True if need to finish the process 
"""
function print_output(goal,goal_checked, output, matrix, type="table")
    if (goal == goal_checked)
        if(isnothing(output))
            if(type == "table")
                pretty_table(matrix, noheader = true, crop = :horizontal)
            else
                pretty_table(matrix, crop = :horizontal)
            end
        else
            if(type == "table")
                CSV.write(output, Tables.table(matrix); delim=",", header=false, transform=(col,val)->(typeof(val)==Float64 ? (@sprintf "%.4f" val) : val))
            else
                CSV.write(output, matrix ; delim=",", header=false)
            end
        end
        return true
    end
    return false
end


"""
Calculate the offset of the matrix given for jacobi
"""
function off(A,n)
    sum =0;
    for i=1:n, j=i+1:n
            sum+=2*(A[i,j]^2)
    end
    return sum
end



"""
The function using jacobian method in order to calculate the eigenvalues and eigenvectors.
"""
function jacobi(A,n)
    off_A = off(A,n)
    v= Matrix{Float64}(I,n,n)
    iter = 0;
    while true
        iter+=1
        i=0
        j=0
        max=-1
        for x=1:n, y=x+1:n
            if(abs(A[x,y])>max)
                i=x
                j=y
                max=abs(A[x,y])
            end
        end
        max == 0 && break

        p = Matrix{Float64}(I,n,n)
        θ = (A[j,j]-A[i,i])/(2*A[i,j])
        sign = 1
        θ<0 && (sign=-1)

        t = sign/(abs(θ)+sqrt((θ^2)+1))
        c = 1/sqrt((t^2)+1)
        s=t*c

        p[i,i]=p[j,j]=c
        p[i,j]=s
        p[j,i]=-s
        v = v*p
        A=transpose(p)*A*p
        off_A_new = off(A,n)
        (off_A-off_A_new <= 0.00001 || iter == 100) && break
        off_A = off_A_new
    end 
    jacob = DataFrame(eigenvalue=[A[x,x] for x in 1:n], eigenvector=[v[:,x] for x in 1:n])
    return jacob
end


main(ARGS...)

