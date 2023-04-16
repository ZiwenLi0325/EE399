# EE399
EE399 Homework submission

Face Image Analysis using Correlation, Eigenvalue Decomposition, and Singular Value Decomposition

Author
[Your Name]

# Abstract
This homework assignment explores the concepts of correlation, eigenvalue decomposition, and singular value decomposition (SVD) for analyzing a dataset of face images. Specifically, we will compute correlation matrices between images and identify the most correlated and uncorrelated images. We will also perform eigenvalue decomposition and SVD on the dataset to find the first eigenvectors and principal component directions, and compare them to determine their similarity. Finally, we will compute the percentage of variance captured by the first six SVD modes and plot these modes.

# I. Introduction and Overview
In this homework assignment, we will be working with a dataset of face images. The dataset contains 39 different faces, with about 65 lighting scenes for each face, resulting in a total of 2414 images. Each image has been downsampled to 32 × 32 pixels and converted into grayscale with values between 0 and 1. The dataset is stored in a matrix X of size 1024 × 2414.

# II. Theoretical Background
Correlation
The correlation between two images can be computed as the dot product between their column vectors. Specifically, the correlation between the j-th and k-th images in the dataset is given by:

$c_{jk} = x_j^T x_k$

where $x_j$ and $x_k$ are the column vectors of images j and k, respectively.

Eigenvalue Decomposition
Eigenvalue decomposition is a method for diagonalizing a square matrix into eigenvectors and eigenvalues. Given a square matrix A, we can find its eigenvectors and eigenvalues by solving the following equation:

$Av = \lambda v$

where $\lambda$ is an eigenvalue and v is the corresponding eigenvector.

Singular Value Decomposition
Singular value decomposition (SVD) is a method for factorizing a matrix into a product of three matrices: U, Σ, and V*. Given a matrix X, we can compute its SVD as follows:

$X = U \Sigma V^*$

where U and V are orthogonal matrices, and Σ is a diagonal matrix containing the singular values of X.

# III. Algorithm Implementation and Development
Part (a)
To compute a 100 × 100 correlation matrix C between the first 100 images in the matrix X, we can use the following code:

css
Copy code
C = np.dot(X[:, :100].T, X[:, :100])
We can then plot the correlation matrix using the pcolor function.

Part (b)
To identify the most highly correlated and uncorrelated images from the correlation matrix C, we can find the maximum and minimum values of the matrix and their corresponding indices. We can then plot these images using the imshow function.

Part (c)
To compute a 10 × 10 correlation matrix between images, we can repeat the procedure in part (a) with the first 10 images in the matrix X.

Part (d)
To find the first six eigenvectors with the largest magnitude eigenvalue, we can create the matrix Y = XXT and compute its eigenvectors and eigenvalues using the eig function.

Part (e)
To find the first six principal component directions using SVD, we can compute the SVD of the matrix X and extract the first six columns of the U matrix.

Part (f)
To compare the first eigenvector v1 from part (d) with the first SVD mode u1 from part (e) and compute the norm of the difference of their absolute values,