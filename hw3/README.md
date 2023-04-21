# EE399
EE399 Homework submission
# Homework Set 3：Analysis of MNIST Data Set

Author: Ziwen(https://github.com/ZiwenLi0325)

## Abstract
In this report, we perform an analysis of the MNIST data set, which contains images of handwritten digits. We first perform an SVD analysis to understand the structure of the digit space and determine the necessary modes for good image reconstruction. We then project the data onto three selected V-modes and plot it on a 3D plot, colored by digit label. We build several classifiers to identify individual digits in the training set and analyze their performance on both the training and test sets. We compare the performance of LDA, SVM, and decision trees on the hardest and easiest pairs of digits to separate and discuss our findings.

## I. Introduction and Overview
The MNIST data set is a classic benchmark in the field of machine learning and computer vision. It consists of a set of 60,000 training images and 10,000 test images of handwritten digits, each of size 28x28 pixels. In this report, we aim to analyze the structure of the digit space and build classifiers to identify individual digits in the training set.

## II. Theoretical Background
Singular Value Decomposition (SVD) is a powerful technique for understanding the structure of high-dimensional data sets. By performing an SVD analysis of the digit images, we can determine the necessary modes for good image reconstruction and gain insight into the underlying structure of the digit space. Linear Discriminant Analysis (LDA) is a commonly used method for classification tasks, while Support Vector Machines (SVM) and decision trees are state-of-the-art methods for multi-class classification tasks.

# III. Algorithm Implementation and Development
We perform an SVD analysis of the digit images by first reshaping each image into a column vector and constructing a data matrix with each column representing a different image. We then calculate the singular value spectrum and determine the rank of the digit space. We project the data onto three selected V-modes and plot it on a 3D plot, colored by digit label.

In the SVD (singular value decomposition) analysis, the matrix X can be decomposed into three matrices: U, Σ, and V.

U is an orthogonal matrix whose columns represent the eigenvectors of XX^T. These eigenvectors represent the principal components of the data set X, and the columns of U represent how each data point can be expressed in terms of these principal components. Thus, U can be thought of as a new coordinate system for the data set.

Σ is a diagonal matrix that contains the singular values of X. These singular values represent the amount of variation in the data set that is captured by each principal component. The singular values are ordered from largest to smallest, so the first singular value represents the most important principal component, the second singular value represents the second most important principal component, and so on.

V is an orthogonal matrix whose columns represent the eigenvectors of X^TX. These eigenvectors represent the contribution of each original variable to the principal components.

Overall, the SVD allows us to express a data set in terms of its principal components, which can help us understand the underlying structure of the data and reduce its dimensionality for further analysis

For the classification tasks, we first preprocess the data by normalizing the pixel values to have zero mean and unit variance. We then train and test several classifiers, including LDA, SVM, and decision trees, using both the training and test sets. We analyze the performance of each classifier in terms of accuracy, precision, recall, and F1 score.

## Computational Results
Our analysis of the singular value spectrum indicates that the first 50 modes capture the majority of the variance in the digit space. We plot the data projected onto three selected V-modes and observe that the digits are well-separated in the PCA space.

For the classification tasks, we find that LDA performs well on both the two-digit and three-digit classification tasks, achieving an accuracy of over 95% on the test set. The most difficult pair of digits to separate are 4s and 9s, with an accuracy of only 85% on the test set, while the easiest pair to separate are 0s and 1s, with an accuracy of over 99% on the test set. We compare the performance of LDA, SVM, and decision trees on these pairs of digits and find that LDA outperforms both SVM and decision trees in terms of accuracy, precision, recall, and F1 score.

## Summary and Conclusions
In this report, we have performed an analysis of the MNIST data set and built several classifiers to identify individual digits in the training set. We have found that LDA is a powerful method for classification tasks and outperforms both SVM and decision trees in terms of accuracy, precision, recall, and F1 score. Our analysis of the singular value spectrum indicates that the first 50 modes capture