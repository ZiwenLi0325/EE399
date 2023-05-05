# EE399
EE399 Homework submission
# Homework Set 4：Neural Network Applications and Comparisons

Author: Ziwen(https://github.com/ZiwenLi0325)

## Abstract:
In this lab report, we examine the application of neural networks to two different datasets. First, we fit a three-layer feed-forward neural network to a given dataset and compare its performance with different training data configurations. Next, we train a feed-forward neural network on the MNIST dataset and compare its performance to other classifiers like LSTM, SVM, and decision trees. The findings demonstrate the potential of neural networks in various scenarios and provide insights into their efficacy compared to other machine learning techniques.

## Sec. I. Introduction and Overview:
In this lab report, we explore the implementation and performance of neural networks in two different scenarios: fitting a given dataset and classifying MNIST digit images. We aim to evaluate the efficiency and accuracy of neural networks and compare their results to other machine learning algorithms.

## Sec. II. Theoretical Background:
Neural networks are computational models inspired by biological neural systems, consisting of interconnected nodes or neurons. These models can learn from data through supervised learning, unsupervised learning, or reinforcement learning. In this lab report, we focus on feed-forward neural networks and supervised learning.

## Sec. III. Algorithm Implementation and Development:
(i) We fit the given dataset to a three-layer feed-forward neural network using the first 20 and first+last 10 data points as training data. We calculate the least-square error for each model over the training points and test data.
(ii) We train a feed-forward neural network on the MNIST dataset, starting by computing the first 20 PCA modes of the digit images. We then build the neural network to classify the digits.

## Sec. IV. Computational Results:
(i) The neural network demonstrates different performance levels when trained on various data configurations. The least-square errors indicate the accuracy of the models. Here are the results for I(i):
```
Epoch [100/1000], Loss: 501.4525
Epoch [200/1000], Loss: 25.1516
Epoch [300/1000], Loss: 13.6244
Epoch [400/1000], Loss: 11.4410
Epoch [500/1000], Loss: 9.5460
Epoch [600/1000], Loss: 7.8256
Epoch [700/1000], Loss: 6.5521
Epoch [800/1000], Loss: 5.7648
Epoch [900/1000], Loss: 5.3244
Epoch [1000/1000], Loss: 5.0801

Training error: 5.1040
Test error: 20.0131
```
For I(iii), the test loss was found to be:
```
Test loss: 12.8579
```
(iv) Comparing the models fit in homework one to the neural networks in (ii) and (iii), we can observe that the neural network trained with the first 20 data points (ii) has a higher test error (20.0131) than the one trained with the first 10 and last 10 data points (iii) with a test loss of 12.8579. This suggests that the model in (iii) performs better on the test data than the model in (ii). When comparing these neural networks to the models in homework one, it is essential to analyze their respective errors, performance, and suitability for the given dataset. Depending on the specific models used in homework one, the neural networks in (ii) and (iii) might show better or worse performance, and a direct comparison would require analyzing each model's errors and complexities.

## Sec. V. Summary and Conclusions:
This lab report highlights the potential of neural networks in fitting datasets and classifying digit images. The findings provide insights into the effectiveness of neural networks in comparison to other machine learning techniques like LSTM, SVM, and decision trees. The results indicate that neural networks offer promising performance in various scenarios, reinforcing their relevance in machine learning applications.