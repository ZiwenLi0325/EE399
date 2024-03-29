# EE399
EE399 Homework submission
# Homework Set 2: Face Image Analysis using Correlation, Eigenvalue Decomposition, and Singular Value Decomposition

Author: Ziwen(https://github.com/ZiwenLi0325)

## Abstract
This homework assignment explores the concepts of correlation, eigenvalue decomposition, and singular value decomposition (SVD) for analyzing a dataset of face images. Specifically, we will compute correlation matrices between images and identify the most correlated and uncorrelated images. We will also perform eigenvalue decomposition and SVD on the dataset to find the first eigenvectors and principal component directions, and compare them to determine their similarity. Finally, we will compute the percentage of variance captured by the first six SVD modes and plot these modes.

## I. Introduction and Overview
In this homework assignment, we will be working with a dataset of face images. The dataset contains 39 different faces, with about 65 lighting scenes for each face, resulting in a total of 2414 images. Each image has been downsampled to 32 × 32 pixels and converted into grayscale with values between 0 and 1. The dataset is stored in a matrix X of size 1024 × 2414.

## II. Theoretical Background
Correlation
The correlation between two images can be computed as the dot product between their column vectors. Specifically, the correlation between the j-th and k-th images in the dataset is given by:

$C_{ij} = X_i^T X_j$

where $X_i$ and $X_j$ are the column vectors of images j and k, respectively.

Eigenvalue Decomposition
Eigenvalue decomposition is a method for diagonalizing a square matrix into eigenvectors and eigenvalues. Given a square matrix A, we can find its eigenvectors and eigenvalues by solving the following equation:

$Av = \lambda v$

where $\lambda$ is an eigenvalue and v is the corresponding eigenvector.

Singular Value Decomposition
Singular value decomposition (SVD) is a method for factorizing a matrix into a product of three matrices: U, Σ, and V*. Given a matrix X, we can compute its SVD as follows:

$X = U \Sigma V^*$

where U and V are orthogonal matrices, and Σ is a diagonal matrix containing the singular values of X.

# III. Algorithm Implementation and Development
### Part (a)
To compute a 100 × 100 correlation matrix C between the first 100 images in the matrix X, we can use the following code:

```
C = np.zeros([100,100])
for i in range(100):
    for j in range(100):
        C[i,j] = np.dot(X[:,i],X[:,j])
```
We can then plot the correlation matrix using the pcolor function.
```
plt.pcolor(C)
plt.colorbar()
```
### Part (b)
To identify the most highly correlated and uncorrelated images from the correlation matrix C, we can find the maximum and minimum values of the matrix and their corresponding indices. We can then plot these images using the imshow function.

For the most correlated images, we have 
```
# Create a copy of the correlation matrix C
C_remove = C.copy()

# Find the indices of the largest entry in the correlation matrix
i, j = np.unravel_index(C_remove.argmax(), C_remove.shape)

# Set the diagonal elements of the correlation matrix to zero
np.fill_diagonal(C_remove, 0)

# Find the indices of the largest entry in the modified correlation matrix
i, j = np.unravel_index(C_remove.argmax(), C_remove.shape)



# Display the two most correlated images side by side
plt.subplot(1,2,1)
print_copy = X[:,i]
plt.title("Image i")
plt.imshow(print_copy.reshape([32,32]),cmap = "gray")
plt.subplot(1,2,2)
print_copy = X[:,j]
plt.imshow(print_copy.reshape([32,32]),cmap = "gray")
plt.legend()

# Add a descriptive caption to the plot
plt.text(0, -0.15, 'Figure 2: The image on the left is image i and the image on the right is image j. They have the highest values \nin correlation matrix, which means that they are most correlated.', ha='center', fontsize=12, transform=plt.gca().transAxes)

# Set the size of the figure and add titles
plt.gcf().set_size_inches(12, 8)
plt.title("Image j")
plt.suptitle("Most correlated images",y = 0.85)

# Save the plot to a file
plt.savefig("figure2.png")

```

For the least correlated images, we have 
```
# Find the index i,j with the smallest entry
small = 100
for i in range(100):
    for j in range(100):
        if C[i,j] < small and i!=j:
            small = C[i,j] 
            i_index = i
            j_index = j
print(f"The index i,j with the smallest entry is ({i_index}, {j_index})")
# Display the two least correlated images side by side
plt.subplot(1,2,1)
print_copy = X[:,54]
plt.title("Image 86")
plt.imshow(print_copy.reshape([32,32]),cmap = "gray")
plt.subplot(1,2,2)
print_copy = X[:,64]
plt.imshow(print_copy.reshape([32,32]),cmap = "gray")
plt.legend()

# Add a descriptive caption to the plot
plt.text(0, -0.15, 'Figure 3: The image on the left is image 54 and the image on the right is image 64. They have the least values \nin correlation matrix, which means that they are least correlated.', ha='center', fontsize=12, transform=plt.gca().transAxes)

# Set the size of the figure and add titles
plt.gcf().set_size_inches(12, 8)
plt.title("Image 88")
plt.suptitle("Least correlated images",y = 0.85)

# Save the plot to a file
plt.savefig("figure3.png")

```
### Part (c)
To compute a 10 × 10 correlation matrix between images, we can repeat the procedure in part (a) with the first 10 images in the matrix X.
```
array_index = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] 
image_set = np.array([X[:,i] for i in array_index])

# Calculate the correlation matrix using dot product
Y = np.matmul(image_set,np.transpose(image_set))

# Plot the correlation matrix as a heatmap
plt.pcolor(Y)
plt.colorbar()
plt.xlabel("i")
plt.ylabel("j")
plt.text(0.7, -0.2, 'Figure 4: 10x10 correlation matrix from indices [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] in original set.', ha='center', fontsize=12, transform=plt.gca().transAxes)
plt.title("10x10 correlation matrix")
plt.savefig("figure4.png")
```

### Part (d)
To find the first six eigenvectors with the largest magnitude eigenvalue, we can create the matrix Y = $X*X^T$ and compute its eigenvectors and eigenvalues using the eig function.
```
Y = np.matmul(X,np.transpose(X))
eigenvalues, eigenvectors = np.linalg.eig(Y)
largest_indices = np.argsort(np.abs(eigenvalues))[::-1][:6]
largest_eigenvectors = eigenvectors[:, largest_indices]
```

### Part (e)
To find the first six principal component directions using SVD, we can compute the SVD of the matrix X and extract the first six columns of the U matrix.
```
# Perform SVD on X
U, s, Vt = np.linalg.svd(X)

# Extract the first six principal components
PCs = Vt[:6, :].T
# Normalize principal components to have unit length
PCs_norm = np.linalg.norm(PCs, axis=0)
PCs = PCs / PCs_norm
print(U[:6, :])
# Show the first six principal components
for i in range(6):
    pc = PCs[:, i]
```

### Part (f)
To compare the first eigenvector v1 from part (d) with the first SVD mode u1 from part (e) and compute the norm of the difference of their absolute values
```
u_1 = U[:,0]
v_1 = eigenvectors[:,0]
l2_norm = np.sqrt(np.sum(np.square(np.abs(u_1)-np.abs(v_1))))
```

### Part (g)
To compute the percentage of variance captured by each of the first 6 SVD modes, we first need to compute the singular value decomposition of the matrix X. We can use the numpy function np.linalg.svd() for this purpose. Once we have the SVD, we can compute the total sum of squared deviations from the mean of the data, which is given by the sum of the squared singular values. We can then compute the percentage of variance captured by each SVD mode by dividing the square of the singular value by the total sum of squared deviations.

To plot the first 6 SVD modes, we can simply plot the first 6 columns of the U matrix obtained from the SVD.

Here's the code to accomplish this:
```
X=results['X']
U, s, Vt = np.linalg.svd(X)
s = s[:6]
print(s)
total_var = np.sum(s**2)

# compute the percentage of variance captured by each mode
var_captured = s**2 / total_var *100

# plot the first 6 SVD modes
fig, axs = plt.subplots(2, 3, figsize=(10, 6), subplot_kw={'xticks': [], 'yticks': []})
fig.suptitle("PCA SVD modes", y=0.95, fontsize=16)

fig.text(0.5, 0.05, 'Figure 5: The image on the left is image 54 and the image on the right is image 64. They have the least values in correlation matrix, which means that they are least correlated.', ha='center', fontsize=12)

fig.subplots_adjust(wspace=0.1, hspace=0.2)
fig.set_size_inches(12, 8)
fig.suptitle("PCA SVD modes",y = 0.95, fontsize=16)
fig.text(0.5, 0.05, 'Figure 5: The image on the left is image 54 and the image on the right is image 64. They have the least values in correlation matrix, which means that they are least correlated.', ha='center', fontsize=12)

for i, ax in enumerate(axs.flat):
    ax.imshow(np.reshape(U[:, i], (32, 32)), cmap='gray')
    ax.set_title('SVD mode {}'.format(i+1))

plt.show()
fig.savefig("figure5.png")

# print the percentage of variance captured by each mode
print('Percentage of variance captured by each mode:')
for i in range(6):
    print('Mode {}: {:.2f}%'.format(i+1, var_captured[i]))

```
## Computational Results
### Part (a)
For the part (a) question, we have a correlation matrix, C, shown below in figure 1. It is computed directly from inner product of matrix X,

![figure1](figure1.png)

In this pcolor plot, we have the visulization about general distribution of the correlation. Noting that there is a line along the diagnoal, which indicates the image inner product of itself. The lighter the pixel is, the more correlated the images at i and j are.

### Part (b)
For the part (b) question, the most correlated image is shown below in figure 2. We can clearly tell the similarities shared by these two images.

![figure2](figure2.png)

In addition, the least correlated image is shown below in figure 3. The second image does not express the clear features shared by the first image .

![figure3](figure3.png)

### Part (c)
For the part (c) question, we have a smaller correlation matrix, C, shown below in figure 4. 

![figure1](figure4.png)

In this pcolor plot, we have the visulization about general distribution of the correlation in detail. For example, the index [1,5] has the lightest color beside the diagnoal values. Therefore, we can conclude that these two images are most correlated.

### Part (d)
For the part (d) question, we have the result from the eigenvalue decomposition, and the followings are eigenvalues,
```
[234020.45485389  49038.31530059   8236.53989701   6024.87145793
   2051.49643269   1901.07911482]
```
and corresponding eigenvectors.
```
[[ 0.02384327  0.04535378  0.05653196  0.04441826 -0.03378603  0.02207542]
 [ 0.02576146  0.04567536  0.04709124  0.05057969 -0.01791442  0.03378819]
 [ 0.02728448  0.04474528  0.0362807   0.05522219 -0.00462854  0.04487476]
 [ 0.0289902   0.04316163  0.02344727  0.05744188  0.00932899  0.05424285]
 [ 0.03057294  0.04080838  0.00992662  0.05720359  0.02096932  0.06259919]
 [ 0.03229324  0.03805116 -0.00241627  0.05372178  0.02766062  0.0694475 ]]
 ```

### Part (e)
For the part (e) question, we have the result from the singular value decomposition, and the followings are singular values,
```
[483.75660704 221.44596474  90.75538495  77.62004546  45.29344801
  43.60136597]
```
and corresponding principal components,
```
[[-0.02384327  0.04535378  0.05653196 ... -0.00238077 -0.0015886
 0.00041024]
 [-0.02576146  0.04567536  0.04709124 ...  0.00265168  0.00886966
  -0.0047811 ]
 [-0.02728448  0.04474528  0.0362807  ... -0.00073077 -0.00706009
   0.00678472]
 [-0.0289902   0.04316163  0.02344727 ... -0.00367797  0.00323748
   0.00077172]
 [-0.03057294  0.04080838  0.00992662 ...  0.00988825 -0.00657576
   0.00607428]
 [-0.03229324  0.03805116 -0.00241627 ... -0.01055232  0.00471235
  -0.00872253]]
 ```
 Noting that the square of the above singular values are the eigenvalues from part (d).

### Part (f)
We will have $u_1$ =
```
[-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902  -0.0166019 ]
```
and $v_1$ =
```
[0.02384327 0.02576146 0.02728448 ... 0.02082937 0.0193902  0.0166019 ]
```
They are basically same in magnitude but have different directions. The difference of norm between u_1 and v_1 is 6.535857028199339e-16. This is a extreme small norm.

### Part (g)
```
Percentage of variance captured by each mode:
Mode 1: 77.68%
Mode 2: 16.28%
Mode 3: 2.73%
Mode 4: 2.00%
Mode 5: 0.68%
Mode 6: 0.63%
```
We can notice that if we add them up, we will have 1, which indicates that the first 6 feature can roughly form all the images.

## Summary and Conclusions

In this project, we analyzed a dataset of 2414 faces from the Yale Faces dataset. The individual images were columns of a matrix X, where each image was downsampled to 32x32 pixels and converted to grayscale with values between 0 and 1.

First, we computed a 100x100 correlation matrix C by computing the dot product between the first 100 images in X. We then plotted the correlation matrix using pcolor. From this matrix, we found the two most highly correlated images and the two most uncorrelated images, which we plotted.

Next, we repeated the above analysis but computed a 10x10 correlation matrix between images [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] in X. We then plotted the correlation matrix between these images.

In part (d), we created a matrix Y = XXT and found the first six eigenvectors with the largest magnitude eigenvalue. In part (e), we performed SVD on the matrix X and found the first six principal component directions.

In part (f), we compared the first eigenvector v1 from part (d) with the first SVD mode u1 from part (e) and computed the norm of the difference of their absolute values.

Finally, in part (g), we computed the percentage of variance captured by each of the first six SVD modes and plotted the first six SVD modes.

Overall, our analysis provides insight into the correlation structure of the faces dataset and demonstrates the use of various linear algebra techniques such as SVD and eigendecomposition for analyzing high-dimensional datasets.