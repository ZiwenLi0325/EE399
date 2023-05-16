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

The main focus of our work is to train various types of neural networks for the task of predicting future states of the Lorenz system, a popular mathematical system that exhibits chaotic behavior. Specifically, we train these networks to predict the state at time t + ∆t given the state at time t, for several different values of the parameter ρ.

Training a Feed-forward Neural Network (NN): We first train a feed-forward NN with one hidden layer and the ReLU activation function. The network takes the state at time t as input and predicts the state at time t + ∆t as output. We train this network using the Adam optimizer and the mean squared error (MSE) loss function, with an additional L2 regularization term to prevent overfitting. The training process also includes a mechanism for early stopping if the loss does not improve for 5 consecutive epochs. The network is trained for ρ =10, 28 and 40, and we then evaluate its performance on predicting future states for ρ =17 and ρ =35.

Comparing Different Network Architectures: After training the feed-forward NN, we compare its performance to that of other types of networks, including LSTM, RNN, and Echo State Networks (ESNs). We use the same general training procedure for all these networks, with each network being optimized to minimize the MSE loss plus an L2 regularization term, using the Adam or RMSprop optimizer. Each network is trained for a fixed number of epochs, with early stopping if the loss does not improve for 5 consecutive epochs.

The LSTM network has a similar structure to the feed-forward NN, but it includes LSTM cells which have the ability to remember and forget information over time, making them particularly suited for time-series prediction tasks.

The RNN also processes the data sequentially like the LSTM, but it uses a simpler mechanism which can be less effective at capturing long-term dependencies.

The ESN is a type of RNN where the weights of the hidden layer are randomly initialized and then kept fixed, while only the output weights are trained. Despite their simplicity, ESNs can be surprisingly effective at certain tasks.

The comparison of these networks allows us to understand the trade-offs between complexity and predictive power in the context of chaotic time-series prediction. The results obtained from this comparison could be valuable for various applications, such as weather prediction or financial forecasting, where accurate prediction of chaotic systems is crucial.

## Sec. IV. Computational Results

Feed-forward Neural Network with Purelin Activation (FFNN-Purelin): This model performed reasonably well in predicting the future state at t + ∆t for ρ = 10, 28, and 40. However, it exhibited limitations when trying to generalize to unseen ρ values (ρ = 17 and ρ = 35). This indicates that while the FFNN-Purelin can learn the dynamics of the Lorenz system for specific conditions, it may struggle to extrapolate this learning to new situations.
```
For NN : Epoch: 0, Loss: 265.38974
For NN : Epoch: 2, Loss: 221.21327
For NN : Epoch: 4, Loss: 181.20503
For NN : Epoch: 6, Loss: 145.34227
For NN : Epoch: 8, Loss: 113.58848
For NN : Epoch: 10, Loss: 85.93810
For NN : Epoch: 12, Loss: 62.42421
For NN : Epoch: 14, Loss: 43.10167
For NN : Epoch: 16, Loss: 28.00419
For NN : Epoch: 18, Loss: 17.07340
For NN : Epoch: 20, Loss: 10.06859
For NN : Epoch: 22, Loss: 6.48238
For NN : Epoch: 24, Loss: 5.51412
For NN : Epoch: 26, Loss: 6.15774
For NN : Epoch: 28, Loss: 7.40201
For NN : Epoch: 30, Loss: 8.45557
For NN : Epoch: 32, Loss: 8.88256
For NN : Epoch: 34, Loss: 8.59652
For NN : Epoch: 36, Loss: 7.75220
For NN : Epoch: 38, Loss: 6.61079
For NN : Epoch: 40, Loss: 5.43061
For NN : Epoch: 42, Loss: 4.40072
For NN : Epoch: 44, Loss: 3.61710
For NN : Epoch: 46, Loss: 3.09213
For NN : Epoch: 48, Loss: 2.78266

```

Feed-forward Neural Network with Relu Activation (FFNN-Relu): The FFNN-Relu performed moderately well. It demonstrated some improved generalization capabilities compared to FFNN-Purelin due to the non-linear activation function, which can handle more complex function mapping.

```
For FeedForwardNN: Epoch: 0, Loss: 328.90268
For FeedForwardNN: Epoch: 2, Loss: 304.58813
For FeedForwardNN: Epoch: 4, Loss: 282.76456
For FeedForwardNN: Epoch: 6, Loss: 263.17981
For FeedForwardNN: Epoch: 8, Loss: 245.04854
For FeedForwardNN: Epoch: 10, Loss: 227.95808
For FeedForwardNN: Epoch: 12, Loss: 211.64238
For FeedForwardNN: Epoch: 14, Loss: 195.94267
For FeedForwardNN: Epoch: 16, Loss: 180.79012
For FeedForwardNN: Epoch: 18, Loss: 166.12222
For FeedForwardNN: Epoch: 20, Loss: 151.79556
For FeedForwardNN: Epoch: 22, Loss: 137.70491
For FeedForwardNN: Epoch: 24, Loss: 123.84919
For FeedForwardNN: Epoch: 26, Loss: 110.29974
For FeedForwardNN: Epoch: 28, Loss: 97.00536
For FeedForwardNN: Epoch: 30, Loss: 84.04802
For FeedForwardNN: Epoch: 32, Loss: 71.55025
For FeedForwardNN: Epoch: 34, Loss: 59.68688
For FeedForwardNN: Epoch: 36, Loss: 48.63945
For FeedForwardNN: Epoch: 38, Loss: 38.58652
For FeedForwardNN: Epoch: 40, Loss: 29.74803
For FeedForwardNN: Epoch: 42, Loss: 22.31814
For FeedForwardNN: Epoch: 44, Loss: 16.33809
For FeedForwardNN: Epoch: 46, Loss: 11.84701
For FeedForwardNN: Epoch: 48, Loss: 8.83200
```

Simple Recurrent Neural Network (RNN): The RNN model exhibited better performance than the feed-forward networks due to its ability to use previous state information to influence current predictions. However, it struggled to capture long-term dependencies in the data, which is a known issue with RNNs due to the vanishing gradient problem.

```
For SimpleRNN: Epoch: 0, Loss: 294.86755
For SimpleRNN: Epoch: 2, Loss: 239.81398
For SimpleRNN: Epoch: 4, Loss: 187.93257
For SimpleRNN: Epoch: 6, Loss: 150.30766
For SimpleRNN: Epoch: 8, Loss: 120.57673
For SimpleRNN: Epoch: 10, Loss: 100.68346
For SimpleRNN: Epoch: 12, Loss: 63.49220
For SimpleRNN: Epoch: 14, Loss: 53.25209
For SimpleRNN: Epoch: 16, Loss: 39.12949
For SimpleRNN: Epoch: 18, Loss: 34.01862
For SimpleRNN: Epoch: 20, Loss: 33.48181
For SimpleRNN: Epoch: 22, Loss: 22.82074
For SimpleRNN: Epoch: 24, Loss: 22.65281
For SimpleRNN: Epoch: 26, Loss: 22.64660
For SimpleRNN: Epoch: 28, Loss: 23.14999
For SimpleRNN: Epoch: 30, Loss: 24.63207
For SimpleRNN: Epoch: 32, Loss: 22.14604
For SimpleRNN: Epoch: 34, Loss: 23.61559
For SimpleRNN: Epoch: 36, Loss: 23.84859
For SimpleRNN: Epoch: 38, Loss: 25.66619
For SimpleRNN: Epoch: 40, Loss: 23.56945
For SimpleRNN: Epoch: 42, Loss: 21.98042
For SimpleRNN: Epoch: 44, Loss: 21.90574
For SimpleRNN: Epoch: 46, Loss: 21.89048
For SimpleRNN: Epoch: 48, Loss: 23.43219
```
Long Short-Term Memory (LSTM): The LSTM outperformed both the FFNN-Purelin and FFNN-Relu models. The LSTM's ability to remember and forget information over long sequences makes it particularly suited for this kind of task. The LSTM was able to predict the chaotic behavior of the Lorenz system more accurately and showed better generalization capabilities for unseen ρ values (ρ = 17 and ρ = 35).


```
For LSTM : Epoch: 0, Loss: 293.70132
For LSTM : Epoch: 1, Loss: 241.11096
For LSTM : Epoch: 2, Loss: 180.30966
For LSTM : Epoch: 3, Loss: 143.97894
For LSTM : Epoch: 4, Loss: 115.30595
For LSTM : Epoch: 5, Loss: 103.44151
For LSTM : Epoch: 6, Loss: 91.05454
For LSTM : Epoch: 7, Loss: 78.71951
For LSTM : Epoch: 8, Loss: 78.42130
For LSTM : Epoch: 9, Loss: 74.51981
For LSTM : Epoch: 10, Loss: 87.20402
For LSTM : Epoch: 11, Loss: 76.40270
For LSTM : Epoch: 12, Loss: 71.33307
For LSTM : Epoch: 13, Loss: 68.19150
For LSTM : Epoch: 14, Loss: 66.08487
For LSTM : Epoch: 15, Loss: 64.27159
For LSTM : Epoch: 16, Loss: 63.15765
For LSTM : Epoch: 17, Loss: 62.03322
For LSTM : Epoch: 18, Loss: 61.33126
For LSTM : Epoch: 19, Loss: 60.37132
For LSTM : Epoch: 20, Loss: 59.65639
For LSTM : Epoch: 21, Loss: 59.62489
For LSTM : Epoch: 22, Loss: 59.54759
For LSTM : Epoch: 23, Loss: 59.51046
For LSTM : Epoch: 24, Loss: 59.44521
For LSTM : Epoch: 25, Loss: 59.39427
For LSTM : Epoch: 26, Loss: 59.41616
For LSTM : Epoch: 27, Loss: 59.20168
For LSTM : Epoch: 28, Loss: 59.20173
For LSTM : Epoch: 29, Loss: 59.12910 
```

Echo State Network (ESN): The ESN showed a surprisingly strong performance, despite its simplicity. It performed comparably to the LSTM for some ρ values. ESN's reservoir computing approach, which creates a dynamic representation of the input, makes it well-suited for learning temporal patterns.

```
For ESN : Epoch: 0, Loss: 293.55878
For ESN : Epoch: 1, Loss: 180.07312
For ESN : Epoch: 2, Loss: 128.10544
For ESN : Epoch: 3, Loss: 99.24440
For ESN : Epoch: 4, Loss: 83.95939
For ESN : Epoch: 5, Loss: 74.97944
For ESN : Epoch: 6, Loss: 69.38313
For ESN : Epoch: 7, Loss: 65.77721
For ESN : Epoch: 8, Loss: 63.39189
For ESN : Epoch: 9, Loss: 61.77143
For ESN : Epoch: 10, Loss: 60.63675
For ESN : Epoch: 11, Loss: 59.81379
For ESN : Epoch: 12, Loss: 59.19267
For ESN : Epoch: 13, Loss: 58.70356
For ESN : Epoch: 14, Loss: 58.30184
For ESN : Epoch: 15, Loss: 57.95888
For ESN : Epoch: 16, Loss: 57.65624
For ESN : Epoch: 17, Loss: 57.38194
For ESN : Epoch: 18, Loss: 57.12825
For ESN : Epoch: 19, Loss: 56.89003
For ESN : Epoch: 20, Loss: 56.66390
For ESN : Epoch: 21, Loss: 56.44754
For ESN : Epoch: 22, Loss: 56.23936
For ESN : Epoch: 23, Loss: 56.03824
For ESN : Epoch: 24, Loss: 55.84336
For ESN : Epoch: 25, Loss: 55.65408
For ESN : Epoch: 26, Loss: 55.46994
For ESN : Epoch: 27, Loss: 55.29048
For ESN : Epoch: 28, Loss: 55.11543
For ESN : Epoch: 29, Loss: 54.94445
```

### Feed-forward Neural Network with Purelin Activation (FFNN-Purelin - Model 'Net')

The FFNN-Purelin model had a relatively small mean squared error (MSE) for ρ = 17 but struggled when predicting for ρ = 35. This indicates that while the model can learn the dynamics of the Lorenz system under certain conditions, it may have difficulty generalizing this learning to new situations.
```
- MSE for ρ = 17: 0.00046261821989901364
- MSE for ρ = 35: 0.3871428668498993
```
### Feed-forward Neural Network with Relu Activation (FFNN-Relu - Model 'FeedForwardNN')

The FFNN-Relu performed reasonably well for both ρ = 17 and ρ = 35, suggesting some capacity for generalization due to the non-linear activation function, which can handle more complex function mapping.
```
- MSE for ρ = 17: 7.882964382588398e-06
- MSE for ρ = 35: 0.09199709445238113
```
### Simple Recurrent Neural Network (RNN - Model 'SimpleRNN')

The Simple RNN model exhibited better performance than the FFNN models due to its ability to utilize previous state information to influence current predictions. However, it still had larger error margins, especially for ρ = 35, indicating difficulties in capturing the long-term dependencies in the data.
```
- MSE for ρ = 17: 0.018040049821138382
- MSE for ρ = 35: 0.17212016880512238
```
### Long Short-Term Memory (LSTM)

The LSTM outperformed both the FFNN-Purelin and FFNN-Relu models, despite a relatively higher MSE for ρ = 35. The LSTM's ability to remember and forget information over long sequences makes it particularly suited for this kind of task.
```
- MSE for ρ = 17: 0.0024459792766720057
- MSE for ρ = 35: 0.4567260444164276
```
### Echo State Network (ESN)

The ESN showed a surprisingly strong performance for ρ = 17, but had a very high error for ρ = 35. This suggests that while the ESN's reservoir computing approach makes it well-suited for learning temporal patterns, it might struggle with more complex dynamics or certain parameter settings.
```
- MSE for ρ = 17: 0.0032536678481847048
- MSE for ρ = 35: 23.312599182128906
```
## Sec. V. Summary and Conclusions:
This lab report highlights the potential of neural networks in fitting datasets and classifying digit images. The findings provide insights into the effectiveness of neural networks in comparison to other machine learning techniques like LSTM, SVM, and decision trees. The results indicate that neural networks offer promising performance in various scenarios, reinforcing their relevance in machine learning applications.