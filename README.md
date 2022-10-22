# spectral-clustering-project
Spectral Clustering implementation in Julia language.
The program is similar to the [Spectral Clustering implementation in C and Python](https://github.com/asafyi/spectral-clustering-project) with few other options and built to fit the result of this project. This is the reason for using python-numpy random for the initial centroids instead of using Julia's random. 
   
Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them.
The technique makes use of the eigenvalues of the similarity matrix of the data points to perform dimensionality reduction before clustering the data - by k-means (in this implementation).  

Various techniques including "Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)" which implemented here are discussed in the paper: [A Tutorial on Spectral Clustering.pdf](A%20Tutorial%20on%20Spectral%20Clustering.pdf)  
The project requirements: [project description.pdf](project%20description.pdf)


## Requirements
 - CSV
 - DataFrames
 - Tables
 - LinearAlgebra
 - Printf
 - PrettyTables
 
 In addition, you need python 3.x with numpy installed.

## Usage
```bash
julia spkmeans.jl <k> <goal> <input_file> <output_file>
```
**k (int, < N):** Number of required clusters. If equal 0, uses the eigengap heuristic algorithm.

**goal (enum):** Can get the following values:
- jacobi: Calculate and output the eigenvalues and eigenvectors of the input matrix.
- wam: Calculate and output the Weighted Adjacency Matrix.
- ddg: Calculate and output the Diagonal Degree Matrix.
- lnorm: Calculate and output the Normalized Graph Laplacian.
- jacobi2: Calculate and output the eigenvalues and eigenvectors of the lnorm matrix.
- sorted: The eigenvalues and eigenvectors from "jacobi2" are sorted according to eigenvalues.
- U: the matrix containing the vectors first k vectors (if k=0 the value of k is beeing calclutated by "Eigengap Heuristic" tenique) from "sorted" as columns.
- T: the matrix is created by renormalizing each of U’s rows to have unit length.
- spk: Perform full spectral clustering and k-means by using kmeans algorithm on matrix T.


**input_file:**    
The path to the Input file, it will contain N data points for all
above goals except Jacobi, in case the goal is Jacobi the input is a symmetric
matrix, the file extension is .txt or .csv.

**output_file (optional):**    
The path that we want for the output file. In case the path is not given, the result will be printed to the console. The file extension can be .txt or .csv.

**For additional information about the original project and how it works use the [project description file](project%20description.pdf).**

