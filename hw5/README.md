# EE399
EE399 Homework submission
# Homework Set 5: Neural Network-based Prediction of Lorenz System Dynamics

Author: Ziwen(https://github.com/ZiwenLi0325)

## Abstract:
This homework explores the application of various neural network architectures in predicting the dynamics of the Lorenz system for different rho values. We train neural networks to advance the solution from time 't' to 't + ∆t' for ρ=10, 28, and 40, and then examine their prediction accuracy for ρ=17 and ρ=35. The performance of feed-forward networks, LSTM, Simple RNN, and Echo State Networks are compared, revealing their distinct strengths and weaknesses in predicting complex, non-linear dynamical systems.

## Sec. I. Introduction and Overview:
In this homework, we delve into the prediction of the Lorenz system dynamics, a classic example of a complex and chaotic system, using various neural network architectures. The objective is to train the models on specific rho values (ρ=10, 28, 40) and evaluate their ability to generalize and accurately predict future states for other rho values (ρ=17, 35). This study will help us understand the robustness and adaptability of different neural network models in the face of non-linear dynamical systems.

## Sec. II. Theoretical Background:
The Lorenz system is a set of three differential equations that describe the chaotic behavior of weather systems. Predicting the Lorenz system dynamics is a challenging task due to its non-linear and chaotic nature.

Neural networks, with their ability to learn from data and model complex relationships, provide a promising approach for such a task. We explore different types of neural networks in this homework:

Feed-forward neural networks: These are the simplest type of artificial neural network. In a feed-forward network, the information moves in only one direction, from the input layer, through the hidden layers, to the output layer. There are no loops in the network.

LSTM (Long Short-Term Memory): LSTMs are a special kind of recurrent neural network capable of learning long-term dependencies. They are particularly good at handling sequences, making them ideal for time series data.

Simple RNN (Recurrent Neural Network): RNNs have loops that allow information to be passed from one step in the sequence to the next. They are particularly useful for time series prediction.

Echo State Networks (ESNs): ESNs are a type of reservoir computing system, which is a neural network with a recurrently connected hidden layer (the "reservoir"). The training only modifies the weights of the readout layer, leaving the reservoir's weights fixed. This makes ESNs relatively simple and efficient for certain tasks.

In this homework, we will train these models and evaluate their effectiveness in predicting the dynamics of the Lorenz system.

## Sec. III. Algorithm Implementation and Development:

## Sec. V. Summary and Conclusions:
This lab report highlights the potential of neural networks in fitting datasets and classifying digit images. The findings provide insights into the effectiveness of neural networks in comparison to other machine learning techniques like LSTM, SVM, and decision trees. The results indicate that neural networks offer promising performance in various scenarios, reinforcing their relevance in machine learning applications.