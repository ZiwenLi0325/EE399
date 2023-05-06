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
### (I)(i)
 We implemented a three-layer feed-forward neural network using the PyTorch library. The network architecture consists of three linear layers with ReLU activation functions for the first two layers. The code for the network is shown below:

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
We defined the input and output data, as well as the loss function (MSE loss), and the optimizer (Adam). The network was trained for 1000 epochs, and the performance was evaluated using the predicted and actual values of the dataset.

### (ii)
We will train a feed-forward neural network on the MNIST dataset. We then build the neural network to classify the digits. The code for training the network on the MNIST dataset is shown below:
```
# Define the training and test data
X_train = torch.tensor(X[:20], dtype=torch.float32).view(-1, 1)
Y_train = torch.tensor(Y[:20], dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X[20:], dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y[20:], dtype=torch.float32).view(-1, 1)

# Initialize the network and define the loss function and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# Train the network
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Compute the predictions on the training and test data
train_predictions = net(X_train)
test_predictions = net(X_test)

# Compute the least square errors on the training and test data
train_error = criterion(train_predictions, Y_train).item()
test_error = criterion(test_predictions, Y_test).item()

print('Training error: {:.4f}'.format(train_error))
print('Test error: {:.4f}'.format(test_error))

```
### (iii)
For the given dataset, we trained the neural network with different data configurations and computed the least square errors on the training and test data. The essential code for training the network and computing the errors is shown below:
```
# Define the training and test data
X_train = torch.tensor(X[:20], dtype=torch.float32).view(-1, 1)
Y_train = torch.tensor(Y[:20], dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X[20:], dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y[20:], dtype=torch.float32).view(-1, 1)

# Initialize the network and define the loss function and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# Train the network
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Compute the predictions on the training and test data
train_predictions = net(X_train)
test_predictions = net(X_test)

# Compute the least square errors on the training and test data
train_error = criterion(train_predictions, Y_train).item()
test_error = criterion(test_predictions, Y_test).item()

print('Training error: {:.4f}'.format(train_error))
print('Test error: {:.4f}'.format(test_error))
```

### (II)
PCA with SVM:
To reduce the dimensionality of the data and potentially improve the performance of the SVM model, we first applied PCA on the training images. We computed the first 20 principal components using the scikit-learn PCA implementation. After transforming the images using the PCA components, we trained a new SVM model using the reduced-dimensionality data and evaluated its performance on the test data. We also printed the explained variance ratio for each principal component.
```
# Load the MNIST dataset and convert to numpy arrays
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_images = train_dataset.data.numpy()
num_samples, num_pixels = train_images.shape[0], train_images.shape[1] * train_images.shape[2]
train_images = train_images.reshape(num_samples, num_pixels)

# Compute the first 20 PCA modes
pca = PCA(n_components=20)
pca.fit(train_images)

# Print the explained variance of each mode
print(pca.explained_variance_ratio_)
```
Neural Network:
We implemented a simple feedforward neural network using the PyTorch library. The network consisted of one hidden layer with 128 units, using the ReLU activation function. The output layer had 10 units, corresponding to the 10 digits, and used the log-softmax activation function. We trained the model using the negative log-likelihood loss and the stochastic gradient descent (SGD) optimizer. The model was trained for 10 epochs, and its performance was evaluated on the test data.
```
# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Instantiate the neural network model and define the loss function and optimizer
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network model
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss/len(train_loader)))

# Evaluate the neural network model on the test data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy: %.3f%%' % (100 * correct / total))
```
LSTM:
We implemented an LSTM-based model using the PyTorch library. The LSTM layer had an input size of 28 (corresponding to the width of the image), a hidden size of 128, and one layer. The output of the LSTM layer was connected to a fully connected layer with 10 output units and a log-softmax activation function. The model was trained using the negative log-likelihood loss and the SGD optimizer, similar to the neural network. The model was trained for 10 epochs, and its performance was evaluated on the test data.
```
# Instantiate the LSTM model and define the loss function and optimizer
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the LSTM model
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss/len(train_loader)))
```

Support Vector Machine (SVM):
We used the scikit-learn library to create a linear SVM model. We first converted the images from the PyTorch DataLoader into a format suitable for scikit-learn. After reshaping the images into 2D arrays, we trained the SVM model using the training data and evaluated its performance on the test data. The accuracy of the SVM model was then calculated.
```
# Preprocess the data for SVM
X_train = []
y_train = []
for images, labels in train_loader:
    images = images.permute(0, 2, 3, 1) # Permute the dimensions to (batch size, height, width, number of channels)
    images = images.reshape(images.shape[0], -1) # Reshape to (batch size, height x width)
    X_train.append(images.numpy())
    y_train.append(labels.numpy())
X_train = torch.from_numpy(np.concatenate(X_train, axis=0))
y_train = torch.from_numpy(np.concatenate(y_train, axis=0))
X_test = []
y_test = []
for images, labels in test_loader:
    images = images.permute(0, 2, 3, 1) # Permute the dimensions to (batch size, height, width, number of channels)
    images = images.reshape(images.shape[0], -1) # Reshape to (batch size, height x width)
    X_test.append(images.numpy())
    y_test.append(labels.numpy())
X_test = torch.from_numpy(np.concatenate(X_test, axis=0))
y_test = torch.from_numpy(np.concatenate(y_test, axis=0))

# Define the SVM model
model = svm.SVC(kernel='linear')

# Train the SVM model
model.fit(X_train, y_train)

# Evaluate the SVM model on the test data
y_pred = model.predict(X_test)
correct = torch.sum(torch.from_numpy(y_pred) == y_test).item()
accuracy = correct / len(y_test)
print('Accuracy: %.3f' % accuracy)
```

Decision Tree Classifier:

We utilized scikit-learn to implement a Decision Tree Classifier for the MNIST dataset. After preprocessing the images from the PyTorch DataLoader into a suitable format, we trained the classifier on the training data and evaluated its performance on the test data. The accuracy of the Decision Tree Classifier was calculated, providing insight into its effectiveness on the MNIST dataset, although more advanced models like CNNs generally outperform it for image classification tasks.
```
# Preprocess the data for the decision tree classifier
X_train = []
y_train = []
for images, labels in train_loader:
    images = images.view(images.shape[0], -1)  # Reshape to (batch size, height x width)
    X_train.append(images.numpy())
    y_train.append(labels.numpy())

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

X_test = []
y_test = []
for images, labels in test_loader:
    images = images.view(images.shape[0], -1)  # Reshape to (batch size, height x width)
    X_test.append(images.numpy())
    y_test.append(labels.numpy())

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the decision tree classifier on the test data
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
```


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
(iv)
For section (ii), we trained the neural network on the first 20 data points and evaluated its performance on the remaining 10 data points. The least squares error for this model was 9.39 on the training data and 16.38 on the test data.

In section (iii), we trained the neural network on a different set of training data, consisting of the first 10 and last 10 data points, and evaluated its performance on the middle 10 data points. The least squares error for this model was 14.46 on the training data and 11.77 on the test data.

Comparing the models fit in homework to the neural networks in (ii) and (iii), we can observe that the neural network trained with the first 20 data points (ii) has a higher test error (20.0131) than the one trained with the first 10 and last 10 data points (iii) with a test loss of 12.8579. This suggests that the model in (iii) performs better on the test data than the model in (ii). This could be attributed to the contiguous nature of the data in section (ii), making it easier for the model to learn the underlying patterns.

On the other hand, the least squares errors for the test data in section (iii) are lower than those in section (ii). This might be due to the more diverse training data used in section (iii), which consists of data points from both the beginning and end of the dataset, potentially providing a better representation of the overall data distribution.

II(i)
The computed result is an array containing the explained variance ratios of the first 20 Principal Component Analysis (PCA) modes for the MNIST dataset. The explained variance ratio represents the proportion of the total variance in the data that is captured by each mode.
```
[0.09704664 0.07095924 0.06169089 0.05389419 0.04868797 0.04312231
 0.0327193  0.02883895 0.02762029 0.02357001 0.02109189 0.02022991
 0.01715814 0.01692108 0.01578641 0.01482947 0.01324532 0.0127688
 0.01187148 0.01152607]
```
II(ii)
The test accuracy of the neural network model is 94.700%. The model was trained for 10 epochs, and the loss was computed after each epoch. The loss decreased with each epoch, indicating that the model was learning the underlying patterns in the data. The loss values for each epoch are shown below:
```
Epoch 1 loss: 0.750
Epoch 2 loss: 0.363
Epoch 3 loss: 0.319
Epoch 4 loss: 0.292
Epoch 5 loss: 0.270
Epoch 6 loss: 0.252
Epoch 7 loss: 0.235
Epoch 8 loss: 0.219
Epoch 9 loss: 0.205
Epoch 10 loss: 0.193
Test accuracy: 94.700%
```
The test accuracy of the LSTM model is 95.270%. The model was trained for 10 epochs, and the loss was computed after each epoch. The loss decreased with each epoch, indicating that the model was learning the underlying patterns in the data. The loss values for each epoch are shown below:
```
Epoch 1 loss: 2.297
Epoch 2 loss: 2.276
Epoch 3 loss: 2.057
Epoch 4 loss: 1.423
Epoch 5 loss: 0.851
Epoch 6 loss: 0.505
Epoch 7 loss: 0.333
Epoch 8 loss: 0.249
Epoch 9 loss: 0.203
Epoch 10 loss: 0.176
Test accuracy: 95.270%
```
The test accuracy of the SVM model is 93.6%. The model was trained on the training data and evaluated on the test data. The accuracy was computed by comparing the predicted labels to the actual labels of the test data.
```
Accuracy: 0.936
```

(ii)
## Sec. V. Summary and Conclusions:
This lab report highlights the potential of neural networks in fitting datasets and classifying digit images. The findings provide insights into the effectiveness of neural networks in comparison to other machine learning techniques like LSTM, SVM, and decision trees. The results indicate that neural networks offer promising performance in various scenarios, reinforcing their relevance in machine learning applications.