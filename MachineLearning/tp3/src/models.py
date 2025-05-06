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

class NeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.1, num_classes=49):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.parameters = self.initialize_parameters()

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
        return np.sum(log_likelihood) / m

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
        grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
        grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

        dA_prev = dZL @ self.parameters[f"W{L}"].T

        for l in reversed(range(1, L)):
            dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])
            grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
            dA_prev = dZ @ self.parameters[f"W{l}"].T

        return grads

    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def train(self, X_train, Y_train, X_val, Y_val, epochs):
        train_losses, val_losses = [], []

        start_time = time.time()

        for epoch in range(epochs):
            Y_pred_train, cache = self.forward(X_train)
            loss_train = self.cross_entropy(Y_pred_train, Y_train)
            grads = self.backward(Y_pred_train, Y_train, cache)
            self.update_parameters(grads)

            Y_pred_val, _ = self.forward(X_val)
            loss_val = self.cross_entropy(Y_pred_val, Y_val)

            train_losses.append(loss_train)
            val_losses.append(loss_val)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

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