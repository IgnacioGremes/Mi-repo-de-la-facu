import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# class NeuralNetwork:
#     def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
#         """
#         Initialize neural network with L hidden layers.
#         hidden_layers: list of integers specifying nodes per hidden layer [M^1, M^2, ..., M^L]
#         """
#         self.learning_rate = learning_rate
#         self.layers = [input_size] + hidden_layers + [output_size]
#         self.weights = []
#         self.biases = []
#         self.activations = []
#         self.train_losses = []  # Store training loss history
#         self.val_losses = []    # Store validation loss history
        
#         # Initialize weights and biases
#         for i in range(len(self.layers)-1):
#             w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
#             b = np.zeros((1, self.layers[i+1]))
#             self.weights.append(w)
#             self.biases.append(b)
    
#     def relu(self, x):
#         """ReLU activation function"""
#         return np.maximum(0, x)
    
#     def relu_derivative(self, x):
#         """Derivative of ReLU"""
#         return np.where(x > 0, 1, 0)
    
#     def softmax(self, x):
#         """Softmax activation function"""
#         exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
#     def cross_entropy_loss(self, y_pred, y_true):
#         """Cross-entropy loss function for one-hot encoded labels"""
#         m = y_true.shape[0]
#         log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
#         return np.sum(log_likelihood) / m
    
#     def accuracy(self, y_pred, y_true):
#         """Compute accuracy for one-hot encoded labels"""
#         predictions = np.argmax(y_pred, axis=1)
#         true_labels = np.argmax(y_true, axis=1)
#         return np.mean(predictions == true_labels)
    
#     def confusion_matrix(self, y_pred, y_true, num_classes):
#         """Compute confusion matrix for one-hot encoded labels"""
#         predictions = np.argmax(y_pred, axis=1)
#         true_labels = np.argmax(y_true, axis=1)
#         cm = np.zeros((num_classes, num_classes), dtype=int)
#         for pred, true in zip(predictions, true_labels):
#             cm[true, pred] += 1
#         return cm
    
#     def forward(self, X):
#         """Forward propagation"""
#         self.activations = [X]
#         current = X

#         # Hidden layers with ReLU
#         for i in range(len(self.weights)-1):
#             z = np.dot(current, self.weights[i]) + self.biases[i]
#             current = self.relu(z)
#             self.activations.append(current)
        
#         # Output layer with softmax
#         z = np.dot(current, self.weights[-1]) + self.biases[-1]
#         y_pred = self.softmax(z)
#         self.activations.append(y_pred)
        
#         return y_pred
    
#     def backward(self, X, y_true, y_pred):
#         """Backward propagation for multi-class classification with one-hot encoded labels"""
#         m = X.shape[0]
#         gradients_w = [np.zeros_like(w) for w in self.weights]
#         gradients_b = [np.zeros_like(b) for b in self.biases]
        
#         # Output layer gradient (softmax + cross-entropy)
#         delta = y_pred - y_true
#         gradients_w[-1] = np.dot(self.activations[-2].T, delta) / m
#         gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
#         # Hidden layers gradients
#         for i in range(len(self.weights)-2, -1, -1):
#             delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
#             gradients_w[i] = np.dot(self.activations[i].T, delta) / m
#             gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
        
#         return gradients_w, gradients_b
    
#     def update_parameters(self, gradients_w, gradients_b):
#         """Update weights and biases using gradient descent"""
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.learning_rate * gradients_w[i]
#             self.biases[i] -= self.learning_rate * gradients_b[i]
    
#     def train(self, X_train, y_train, X_val, y_val, epochs, num_classes):
#         """Train the neural network and compute metrics"""
        
#         for epoch in range(epochs):
#             # Forward pass on training data
#             y_pred_train = self.forward(X_train)
            
#             # Backward pass
#             grad_w, grad_b = self.backward(X_train, y_train, y_pred_train)
            
#             # Update parameters
#             self.update_parameters(grad_w, grad_b)
            
#             # Compute training loss and accuracy
#             train_loss = self.cross_entropy_loss(y_pred_train, y_train)
#             train_acc = self.accuracy(y_pred_train, y_train)
#             self.train_losses.append(train_loss)
            
#             # Compute validation loss and accuracy
#             y_pred_val = self.forward(X_val)
#             val_loss = self.cross_entropy_loss(y_pred_val, y_val)
#             val_acc = self.accuracy(y_pred_val, y_val)
#             self.val_losses.append(val_loss)
            
#             # if epoch % 10 == 0:
#             #     print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#             #           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
#         # Final metrics
#         print("\nFinal Metrics:")
#         print("Training Set:")
#         print(f"Accuracy: {train_acc:.4f}")
#         print(f"Cross-Entropy Loss: {train_loss:.4f}")
#         print("Confusion Matrix:")
#         cm_train = self.confusion_matrix(y_pred_train, y_train, num_classes)
#         print(cm_train)
        
#         print("\nValidation Set:")
#         print(f"Accuracy: {val_acc:.4f}")
#         print(f"Cross-Entropy Loss: {val_loss:.4f}")
#         print("Confusion Matrix:")
#         cm_val = self.confusion_matrix(y_pred_val, y_val, num_classes)
#         print(cm_val)
    
#     def plot_losses(self, epochs):
#         """Plot training and validation loss"""
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(epochs), self.train_losses, label='Training Loss')
#         plt.plot(range(epochs), self.val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Cross-Entropy Loss')
#         plt.title('Training and Validation Loss vs. Epoch')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

import time







# class NeuralNetwork:
#     def __init__(self, layer_dims, learning_rate=0.1, num_classes=49):
#         self.layer_dims = layer_dims
#         self.learning_rate = learning_rate
#         self.num_classes = num_classes
#         self.parameters = self.initialize_parameters()

#     def relu(self, Z):
#         return np.maximum(0, Z)

#     def relu_derivative(self, Z):
#         return (Z > 0).astype(float)

#     def softmax(self, Z):
#         expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
#         return expZ / np.sum(expZ, axis=1, keepdims=True)

#     def cross_entropy(self, Y_pred, Y_true):
#         m = Y_true.shape[0]
#         log_likelihood = -np.log(Y_pred[range(m), Y_true])
#         return np.sum(log_likelihood) / m

#     def initialize_parameters(self):
#         np.random.seed(0)
#         parameters = {}
#         for l in range(1, len(self.layer_dims)):
#             parameters[f"W{l}"] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * np.sqrt(2. / self.layer_dims[l-1])
#             parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))
#         return parameters

#     def forward(self, X):
#         cache = {"A0": X}
#         L = len(self.parameters) // 2
#         for l in range(1, L):
#             Z = cache[f"A{l-1}"] @ self.parameters[f"W{l}"] + self.parameters[f"b{l}"]
#             cache[f"Z{l}"] = Z
#             cache[f"A{l}"] = self.relu(Z)
#         ZL = cache[f"A{L-1}"] @ self.parameters[f"W{L}"] + self.parameters[f"b{L}"]
#         AL = self.softmax(ZL)
#         cache[f"Z{L}"] = ZL
#         cache[f"A{L}"] = AL
#         return AL, cache

#     def backward(self, Y_pred, Y_true, cache):
#         grads = {}
#         m = Y_true.shape[0]
#         L = len(self.parameters) // 2

#         Y_one_hot = np.zeros_like(Y_pred)
#         Y_one_hot[np.arange(m), Y_true] = 1

#         dZL = (Y_pred - Y_one_hot) / m
#         grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
#         grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

#         dA_prev = dZL @ self.parameters[f"W{L}"].T

#         for l in reversed(range(1, L)):
#             dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
#             grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
#             grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
#             dA_prev = dZ @ self.parameters[f"W{l}"].T

#         return grads

#     def update_parameters(self, grads):
#         L = len(self.parameters) // 2
#         for l in range(1, L+1):
#             self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
#             self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

#     def train(self, X_train, Y_train, X_val, Y_val, epochs):
#         train_losses, val_losses = [], []

#         start_time = time.time()

#         for epoch in range(epochs):
#             Y_pred_train, cache = self.forward(X_train)
#             loss_train = self.cross_entropy(Y_pred_train, Y_train)
#             grads = self.backward(Y_pred_train, Y_train, cache)
#             self.update_parameters(grads)

#             Y_pred_val, _ = self.forward(X_val)
#             loss_val = self.cross_entropy(Y_pred_val, Y_val)

#             train_losses.append(loss_train)
#             val_losses.append(loss_val)

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

#         end_time = time.time()
#         print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

#         return train_losses, val_losses

#     def compute_accuracy(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         return np.mean(Y_pred == Y_true)

#     def compute_confusion_matrix(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
#         for t, p in zip(Y_true, Y_pred):
#             conf_matrix[t, p] += 1
#         return conf_matrix

#     def plot_confusion_matrix(self, conf_matrix):
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.title("Confusion Matrix Heatmap")
#         plt.show()

#     def evaluate(self, X, Y):
#         Y_pred_probs, _ = self.forward(X)
#         acc = self.compute_accuracy(Y_pred_probs, Y)
#         loss = self.cross_entropy(Y_pred_probs, Y)
#         conf_matrix = self.compute_confusion_matrix(Y_pred_probs, Y)
#         return acc, loss, conf_matrix

#     def plot_losses(self, train_losses, val_losses):
#         plt.plot(train_losses, label='Train Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Cross-Entropy Loss')
#         plt.legend()
#         plt.title('Loss over Epochs')
#         plt.grid(True)
#         plt.show()







# class NeuralNetwork:
#     def __init__(self, layer_dims, learning_rate=0.1, num_classes=49, schedule=None):
#         self.layer_dims = layer_dims
#         self.learning_rate = learning_rate
#         self.initial_learning_rate = learning_rate
#         self.num_classes = num_classes
#         self.schedule = schedule
#         self.parameters = self.initialize_parameters()

#     def relu(self, Z):
#         return np.maximum(0, Z)

#     def relu_derivative(self, Z):
#         return (Z > 0).astype(float)

#     def softmax(self, Z):
#         expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
#         return expZ / np.sum(expZ, axis=1, keepdims=True)

#     def cross_entropy(self, Y_pred, Y_true):
#         m = Y_true.shape[0]
#         log_likelihood = -np.log(Y_pred[range(m), Y_true])
#         return np.sum(log_likelihood) / m

#     def initialize_parameters(self):
#         np.random.seed(0)
#         parameters = {}
#         for l in range(1, len(self.layer_dims)):
#             parameters[f"W{l}"] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * np.sqrt(2. / self.layer_dims[l-1])
#             parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))
#         return parameters

#     def forward(self, X):
#         cache = {"A0": X}
#         L = len(self.parameters) // 2
#         for l in range(1, L):
#             Z = cache[f"A{l-1}"] @ self.parameters[f"W{l}"] + self.parameters[f"b{l}"]
#             cache[f"Z{l}"] = Z
#             cache[f"A{l}"] = self.relu(Z)
#         ZL = cache[f"A{L-1}"] @ self.parameters[f"W{L}"] + self.parameters[f"b{L}"]
#         AL = self.softmax(ZL)
#         cache[f"Z{L}"] = ZL
#         cache[f"A{L}"] = AL
#         return AL, cache

#     def backward(self, Y_pred, Y_true, cache):
#         grads = {}
#         m = Y_true.shape[0]
#         L = len(self.parameters) // 2

#         Y_one_hot = np.zeros_like(Y_pred)
#         Y_one_hot[np.arange(m), Y_true] = 1

#         dZL = (Y_pred - Y_one_hot) / m
#         grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
#         grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

#         dA_prev = dZL @ self.parameters[f"W{L}"].T

#         for l in reversed(range(1, L)):
#             dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
#             grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
#             grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
#             dA_prev = dZ @ self.parameters[f"W{l}"].T

#         return grads

#     def update_parameters(self, grads):
#         L = len(self.parameters) // 2
#         for l in range(1, L+1):
#             self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
#             self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

#     def apply_learning_rate_schedule(self, epoch):
#         if self.schedule == "linear":
#             self.learning_rate = max(self.initial_learning_rate * (1 - epoch / 100), 0.001)
#         elif self.schedule == "exponential":
#             self.learning_rate = self.initial_learning_rate * (0.95 ** epoch)

#     def train(self, X_train, Y_train, X_val, Y_val, epochs):
#         train_losses, val_losses = [], []

#         start_time = time.time()

#         for epoch in range(epochs):
#             self.apply_learning_rate_schedule(epoch)

#             Y_pred_train, cache = self.forward(X_train)
#             loss_train = self.cross_entropy(Y_pred_train, Y_train)
#             grads = self.backward(Y_pred_train, Y_train, cache)
#             self.update_parameters(grads)

#             Y_pred_val, _ = self.forward(X_val)
#             loss_val = self.cross_entropy(Y_pred_val, Y_val)

#             train_losses.append(loss_train)
#             val_losses.append(loss_val)

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}, LR = {self.learning_rate:.6f}")

#         end_time = time.time()
#         print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

#         return train_losses, val_losses

#     def compute_accuracy(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         return np.mean(Y_pred == Y_true)

#     def compute_confusion_matrix(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
#         for t, p in zip(Y_true, Y_pred):
#             conf_matrix[t, p] += 1
#         return conf_matrix

#     def plot_confusion_matrix(self, conf_matrix):
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.title("Confusion Matrix Heatmap")
#         plt.show()

#     def evaluate(self, X, Y):
#         Y_pred_probs, _ = self.forward(X)
#         acc = self.compute_accuracy(Y_pred_probs, Y)
#         loss = self.cross_entropy(Y_pred_probs, Y)
#         conf_matrix = self.compute_confusion_matrix(Y_pred_probs, Y)
#         return acc, loss, conf_matrix

#     def plot_losses(self, train_losses, val_losses):
#         plt.plot(train_losses, label='Train Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Cross-Entropy Loss')
#         plt.legend()
#         plt.title('Loss over Epochs')
#         plt.grid(True)
#         plt.show()






# class NeuralNetwork:
#     def __init__(self, layer_dims, learning_rate=0.1, num_classes=49, schedule=None, use_mini_batch=False, batch_size=32):
#         self.layer_dims = layer_dims
#         self.learning_rate = learning_rate
#         self.initial_learning_rate = learning_rate
#         self.num_classes = num_classes
#         self.schedule = schedule
#         self.use_mini_batch = use_mini_batch
#         self.batch_size = batch_size
#         self.parameters = self.initialize_parameters()

#     def relu(self, Z):
#         return np.maximum(0, Z)

#     def relu_derivative(self, Z):
#         return (Z > 0).astype(float)

#     def softmax(self, Z):
#         expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
#         return expZ / np.sum(expZ, axis=1, keepdims=True)

#     def cross_entropy(self, Y_pred, Y_true):
#         m = Y_true.shape[0]
#         log_likelihood = -np.log(Y_pred[range(m), Y_true] + 1e-15)
#         return np.sum(log_likelihood) / m

#     def initialize_parameters(self):
#         np.random.seed(0)
#         parameters = {}
#         for l in range(1, len(self.layer_dims)):
#             parameters[f"W{l}"] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * np.sqrt(2. / self.layer_dims[l-1])
#             parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))
#         return parameters

#     def forward(self, X):
#         cache = {"A0": X}
#         L = len(self.parameters) // 2
#         for l in range(1, L):
#             Z = cache[f"A{l-1}"] @ self.parameters[f"W{l}"] + self.parameters[f"b{l}"]
#             cache[f"Z{l}"] = Z
#             cache[f"A{l}"] = self.relu(Z)
#         ZL = cache[f"A{L-1}"] @ self.parameters[f"W{L}"] + self.parameters[f"b{L}"]
#         AL = self.softmax(ZL)
#         cache[f"Z{L}"] = ZL
#         cache[f"A{L}"] = AL
#         return AL, cache

#     def backward(self, Y_pred, Y_true, cache):
#         grads = {}
#         m = Y_true.shape[0]
#         L = len(self.parameters) // 2

#         Y_one_hot = np.zeros_like(Y_pred)
#         Y_one_hot[np.arange(m), Y_true] = 1

#         dZL = (Y_pred - Y_one_hot) / m
#         grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
#         grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

#         dA_prev = dZL @ self.parameters[f"W{L}"].T

#         for l in reversed(range(1, L)):
#             dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
#             grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
#             grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
#             dA_prev = dZ @ self.parameters[f"W{l}"].T

#         return grads

#     def update_parameters(self, grads):
#         L = len(self.parameters) // 2
#         for l in range(1, L+1):
#             self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
#             self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

#     def apply_learning_rate_schedule(self, epoch, epochs):
#         if self.schedule == "linear":
#             self.learning_rate = max(self.initial_learning_rate * (1 - epoch / epochs), 0.001)
#         elif self.schedule == "exponential":
#             self.learning_rate = self.initial_learning_rate * (0.90 ** epoch)

#     def train(self, X_train, Y_train, X_val, Y_val, epochs):
#         train_losses, val_losses = [], []

#         start_time = time.time()

#         for epoch in range(epochs):
#             self.apply_learning_rate_schedule(epoch, epochs)

#             if self.use_mini_batch:
#                 indices = np.random.permutation(X_train.shape[0])
#                 X_train_shuffled = X_train[indices]
#                 Y_train_shuffled = Y_train[indices]

#                 for i in range(0, X_train.shape[0], self.batch_size):
#                     X_batch = X_train_shuffled[i:i+self.batch_size]
#                     Y_batch = Y_train_shuffled[i:i+self.batch_size]
#                     Y_pred_batch, cache = self.forward(X_batch)
#                     grads = self.backward(Y_pred_batch, Y_batch, cache)
#                     self.update_parameters(grads)

#                 Y_pred_train, _ = self.forward(X_train)
#                 loss_train = self.cross_entropy(Y_pred_train, Y_train)
#             else:
#                 Y_pred_train, cache = self.forward(X_train)
#                 loss_train = self.cross_entropy(Y_pred_train, Y_train)
#                 grads = self.backward(Y_pred_train, Y_train, cache)
#                 self.update_parameters(grads)

#             Y_pred_val, _ = self.forward(X_val)
#             loss_val = self.cross_entropy(Y_pred_val, Y_val)

#             train_losses.append(loss_train)
#             val_losses.append(loss_val)

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}, LR = {self.learning_rate:.6f}")

#         end_time = time.time()
#         print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

#         return train_losses, val_losses

#     def compute_accuracy(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         return np.mean(Y_pred == Y_true)

#     def compute_confusion_matrix(self, Y_pred_probs, Y_true):
#         Y_pred = np.argmax(Y_pred_probs, axis=1)
#         conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
#         for t, p in zip(Y_true, Y_pred):
#             conf_matrix[t, p] += 1
#         return conf_matrix

#     def plot_confusion_matrix(self, conf_matrix):
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.title("Confusion Matrix Heatmap")
#         plt.show()

#     def evaluate(self, X, Y):
#         Y_pred_probs, _ = self.forward(X)
#         acc = self.compute_accuracy(Y_pred_probs, Y)
#         loss = self.cross_entropy(Y_pred_probs, Y)
#         conf_matrix = self.compute_confusion_matrix(Y_pred_probs, Y)
#         return acc, loss, conf_matrix

#     def plot_losses(self, train_losses, val_losses):
#         plt.plot(train_losses, label='Train Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Cross-Entropy Loss')
#         plt.legend()
#         plt.title('Loss over Epochs')
#         plt.grid(True)
#         plt.show()








class NeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.1, num_classes=49, schedule=None, use_mini_batch=False, batch_size=32, optimizer='gd', l2_lambda=0.0):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.num_classes = num_classes
        self.schedule = schedule
        self.use_mini_batch = use_mini_batch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.parameters = self.initialize_parameters()
        self.l2_lambda = l2_lambda
        if self.optimizer == 'adam':
            self.v = {}
            self.s = {}
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0
            for l in range(1, len(self.layer_dims)):
                self.v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
                self.v[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])
                self.s[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
                self.s[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def cross_entropy(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_likelihood = -np.log(Y_pred[range(m), Y_true])
        base_loss = np.sum(log_likelihood) / m

        if self.l2_lambda > 0:
            L = len(self.parameters) // 2
            l2_sum = sum(np.sum(np.square(self.parameters[f"W{l}"])) for l in range(1, L + 1))
            l2_penalty = (self.l2_lambda / (2 * m)) * l2_sum
            return base_loss + l2_penalty

        return base_loss

    def initialize_parameters(self):
        np.random.seed(0)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f"W{l}"] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))
        return parameters

    def forward(self, X):
        cache = {"A0": X}
        L = len(self.parameters) // 2
        for l in range(1, L):
            Z = cache[f"A{l-1}"] @ self.parameters[f"W{l}"] + self.parameters[f"b{l}"]
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = self.relu(Z)
        ZL = cache[f"A{L-1}"] @ self.parameters[f"W{L}"] + self.parameters[f"b{L}"]
        AL = self.softmax(ZL)
        cache[f"Z{L}"] = ZL
        cache[f"A{L}"] = AL
        return AL, cache

    def backward(self, Y_pred, Y_true, cache):
        grads = {}
        m = Y_true.shape[0]
        L = len(self.parameters) // 2

        Y_one_hot = np.zeros_like(Y_pred)
        Y_one_hot[np.arange(m), Y_true] = 1

        dZL = (Y_pred - Y_one_hot) / m
        grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL + (self.l2_lambda / m) * self.parameters[f"W{L}"]
        grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

        dA_prev = dZL @ self.parameters[f"W{L}"].T

        for l in reversed(range(1, L)):
            dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
            grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ + (self.l2_lambda / m) * self.parameters[f"W{l}"]
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
            dA_prev = dZ @ self.parameters[f"W{l}"].T

        return grads

    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        if self.optimizer == 'gd':
            for l in range(1, L+1):
                self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
                self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]
        elif self.optimizer == 'adam':
            self.t += 1
            for l in range(1, L+1):
                self.v[f"dW{l}"] = self.beta1 * self.v[f"dW{l}"] + (1 - self.beta1) * grads[f"dW{l}"]
                self.v[f"db{l}"] = self.beta1 * self.v[f"db{l}"] + (1 - self.beta1) * grads[f"db{l}"]
                self.s[f"dW{l}"] = self.beta2 * self.s[f"dW{l}"] + (1 - self.beta2) * (grads[f"dW{l}"] ** 2)
                self.s[f"db{l}"] = self.beta2 * self.s[f"db{l}"] + (1 - self.beta2) * (grads[f"db{l}"] ** 2)

                v_corrected_dw = self.v[f"dW{l}"] / (1 - self.beta1 ** self.t)
                v_corrected_db = self.v[f"db{l}"] / (1 - self.beta1 ** self.t)
                s_corrected_dw = self.s[f"dW{l}"] / (1 - self.beta2 ** self.t)
                s_corrected_db = self.s[f"db{l}"] / (1 - self.beta2 ** self.t)

                self.parameters[f"W{l}"] -= self.learning_rate * v_corrected_dw / (np.sqrt(s_corrected_dw) + self.epsilon)
                self.parameters[f"b{l}"] -= self.learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + self.epsilon)

    def apply_learning_rate_schedule(self, epoch, epochs):
        if self.schedule == "linear":
            self.learning_rate = max(self.initial_learning_rate * (1 - epoch / epochs), 0.001)
        elif self.schedule == "exponential":
            self.learning_rate = self.initial_learning_rate * (0.95 ** epoch)

    def train(self, X_train, Y_train, X_val, Y_val, epochs, early_stopping=False, patience=10, min_delta=1e-4):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        wait = 0  # For early stopping

        start_time = time.time()

        for epoch in range(epochs):
            self.apply_learning_rate_schedule(epoch, epochs)

            if self.use_mini_batch:
                indices = np.random.permutation(X_train.shape[0])
                X_train_shuffled = X_train[indices]
                Y_train_shuffled = Y_train[indices]
                for i in range(0, X_train.shape[0], self.batch_size):
                    X_batch = X_train_shuffled[i:i+self.batch_size]
                    Y_batch = Y_train_shuffled[i:i+self.batch_size]
                    Y_pred, cache = self.forward(X_batch)
                    grads = self.backward(Y_pred, Y_batch, cache)
                    self.update_parameters(grads)
                Y_pred_train, _ = self.forward(X_train)
                loss_train = self.cross_entropy(Y_pred_train, Y_train)
            else:
                Y_pred_train, cache = self.forward(X_train)
                loss_train = self.cross_entropy(Y_pred_train, Y_train)
                grads = self.backward(Y_pred_train, Y_train, cache)
                self.update_parameters(grads)

            Y_pred_val, _ = self.forward(X_val)
            loss_val = self.cross_entropy(Y_pred_val, Y_val)

            train_losses.append(loss_train)
            val_losses.append(loss_val)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}, LR = {self.learning_rate:.6f}")

            # --- EARLY STOPPING BLOCK ---
            if early_stopping:
                if best_val_loss - loss_val > min_delta:
                    best_val_loss = loss_val
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"\nEarly stopping at epoch {epoch} (no improvement in {patience} epochs).")
                        break
            # ----------------------------

        end_time = time.time()
        print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

        return train_losses, val_losses

    def compute_accuracy(self, Y_pred_probs, Y_true):
        Y_pred = np.argmax(Y_pred_probs, axis=1)
        return np.mean(Y_pred == Y_true)

    def compute_confusion_matrix(self, Y_pred_probs, Y_true):
        Y_pred = np.argmax(Y_pred_probs, axis=1)
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(Y_true, Y_pred):
            conf_matrix[t, p] += 1
        return conf_matrix

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix Heatmap")
        plt.show()

    def evaluate(self, X, Y):
        Y_pred_probs, _ = self.forward(X)
        acc = self.compute_accuracy(Y_pred_probs, Y)
        loss = self.cross_entropy(Y_pred_probs, Y)
        conf_matrix = self.compute_confusion_matrix(Y_pred_probs, Y)
        return acc, loss, conf_matrix

    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.grid(True)
        plt.show()













# class NeuralNetwork:
#     def __init__(self, layers, lr=0.01, epochs=100, batch_size=None, 
#                  optimizer='sgd', l2_lambda=0.0, lr_schedule=None):
#         self.layers = layers  # List like [input_dim, hidden1, hidden2, ..., output_dim]
#         self.lr = lr
#         self.initial_lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.optimizer = optimizer.lower()
#         self.l2_lambda = l2_lambda
#         self.lr_schedule = lr_schedule
#         self._init_weights()

#     def _init_weights(self):
#         self.weights = []
#         self.biases = []
#         self.v_dw = []
#         self.v_db = []
#         self.m_dw = []
#         self.m_db = []

#         for i in range(len(self.layers)-1):
#             w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2. / self.layers[i])
#             b = np.zeros((1, self.layers[i+1]))
#             self.weights.append(w)
#             self.biases.append(b)
#             self.v_dw.append(np.zeros_like(w))
#             self.v_db.append(np.zeros_like(b))
#             self.m_dw.append(np.zeros_like(w))
#             self.m_db.append(np.zeros_like(b))

#         self.beta1 = 0.9
#         self.beta2 = 0.999
#         self.epsilon = 1e-8

#     def _relu(self, z):
#         return np.maximum(0, z)

#     def _relu_derivative(self, z):
#         return z > 0

#     def _softmax(self, z):
#         exp = np.exp(z - np.max(z, axis=1, keepdims=True))
#         return exp / np.sum(exp, axis=1, keepdims=True)

#     def _cross_entropy(self, y_true, y_pred):
#         m = y_true.shape[0]
#         loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
#         reg_term = self.l2_lambda * sum([np.sum(w ** 2) for w in self.weights]) / (2 * m)
#         return loss + reg_term

#     def _forward(self, X):
#         activations = [X]
#         zs = []
#         for i in range(len(self.weights)-1):
#             z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
#             a = self._relu(z)
#             zs.append(z)
#             activations.append(a)

#         z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
#         a = self._softmax(z)
#         zs.append(z)
#         activations.append(a)

#         return activations, zs

#     def _backward(self, activations, zs, y_true):
#         grads_w = [0] * len(self.weights)
#         grads_b = [0] * len(self.biases)
#         m = y_true.shape[0]

#         delta = activations[-1] - y_true
#         grads_w[-1] = np.dot(activations[-2].T, delta) / m + self.l2_lambda * self.weights[-1] / m
#         grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

#         for l in range(len(self.weights)-2, -1, -1):
#             delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(zs[l])
#             grads_w[l] = np.dot(activations[l].T, delta) / m + self.l2_lambda * self.weights[l] / m
#             grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m

#         return grads_w, grads_b

#     def _update_params(self, grads_w, grads_b, t):
#         for i in range(len(self.weights)):
#             if self.optimizer == 'adam':
#                 self.m_dw[i] = self.beta1 * self.m_dw[i] + (1 - self.beta1) * grads_w[i]
#                 self.v_dw[i] = self.beta2 * self.v_dw[i] + (1 - self.beta2) * (grads_w[i] ** 2)

#                 self.m_db[i] = self.beta1 * self.m_db[i] + (1 - self.beta1) * grads_b[i]
#                 self.v_db[i] = self.beta2 * self.v_db[i] + (1 - self.beta2) * (grads_b[i] ** 2)

#                 m_dw_hat = self.m_dw[i] / (1 - self.beta1 ** t)
#                 v_dw_hat = self.v_dw[i] / (1 - self.beta2 ** t)
#                 m_db_hat = self.m_db[i] / (1 - self.beta1 ** t)
#                 v_db_hat = self.v_db[i] / (1 - self.beta2 ** t)

#                 self.weights[i] -= self.lr * m_dw_hat / (np.sqrt(v_dw_hat) + self.epsilon)
#                 self.biases[i] -= self.lr * m_db_hat / (np.sqrt(v_db_hat) + self.epsilon)

#             else:
#                 self.weights[i] -= self.lr * grads_w[i]
#                 self.biases[i] -= self.lr * grads_b[i]

#     def _adjust_lr(self, epoch):
#         if self.lr_schedule == 'linear':
#             self.lr = max(1e-4, self.initial_lr * (1 - epoch / self.epochs))
#         elif self.lr_schedule == 'exponential':
#             decay_rate = 0.95
#             self.lr = self.initial_lr * (decay_rate ** epoch)

#     def _one_hot(self, y):
#         classes = np.max(y) + 1
#         return np.eye(classes)[y]

#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         if y_train.ndim == 1:
#             y_train = self._one_hot(y_train)
#         if y_val is not None and y_val.ndim == 1:
#             y_val = self._one_hot(y_val)

#         train_losses = []
#         val_losses = []
#         t = 1
#         start_time = time.time()

#         for epoch in range(1, self.epochs + 1):
#             self._adjust_lr(epoch)

#             if self.batch_size:
#                 indices = np.random.permutation(X_train.shape[0])
#                 for start in range(0, X_train.shape[0], self.batch_size):
#                     end = start + self.batch_size
#                     X_batch = X_train[indices[start:end]]
#                     y_batch = y_train[indices[start:end]]
#                     a, z = self._forward(X_batch)
#                     grads_w, grads_b = self._backward(a, z, y_batch)
#                     self._update_params(grads_w, grads_b, t)
#                     t += 1
#             else:
#                 a, z = self._forward(X_train)
#                 grads_w, grads_b = self._backward(a, z, y_train)
#                 self._update_params(grads_w, grads_b, t)
#                 t += 1

#             a_train, _ = self._forward(X_train)
#             train_loss = self._cross_entropy(y_train, a_train[-1])
#             train_losses.append(train_loss)

#             if X_val is not None and y_val is not None:
#                 a_val, _ = self._forward(X_val)
#                 val_loss = self._cross_entropy(y_val, a_val[-1])
#                 val_losses.append(val_loss)

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {self.lr:.6f}")

#         end_time = time.time()
#         print(f"Training time: {end_time - start_time:.2f} seconds")
#         return train_losses, val_losses

#     def predict(self, X):
#         a, _ = self._forward(X)
#         return np.argmax(a[-1], axis=1)

#     def evaluate(self, X, y_true):
#         if y_true.ndim == 2:
#             y_true_labels = np.argmax(y_true, axis=1)
#         else:
#             y_true_labels = y_true

#         a, _ = self._forward(X)
#         y_pred = np.argmax(a[-1], axis=1)

#         acc = np.sum(y_pred == y_true_labels) / len(y_true_labels)
#         loss = self._cross_entropy(self._one_hot(y_true_labels), a[-1])

#         print(f"Accuracy: {acc:.4f}")
#         print(f"Cross-Entropy Loss: {loss:.4f}")

#         num_classes = np.max(y_true_labels) + 1
#         cm = np.zeros((num_classes, num_classes), dtype=int)
#         for t, p in zip(y_true_labels, y_pred):
#             cm[t, p] += 1

#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title('Confusion Matrix')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.show()

#         return acc, loss

#     def plot_loss(self, train_losses, val_losses):
#         plt.plot(train_losses, label='Train Loss')
#         if val_losses:
#             plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Cross-Entropy Loss')
#         plt.title('Loss Curve')
#         plt.legend()
#         plt.grid(True)
#         plt.show()