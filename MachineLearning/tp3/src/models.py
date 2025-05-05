import numpy as np

class NeuralNetwork_M0:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        """
        Initialize neural network with L hidden layers.
        hidden_layers: list of integers specifying nodes per hidden layer [M^1, M^2, ..., M^L]
        """
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activations = []
        self.losses = []
        
        # Initialize weights and biases
        for i in range(len(self.layers)-1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        """Cross-entropy loss function"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        return np.sum(log_likelihood) / m
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        current = X
        
        # Hidden layers with ReLU
        for i in range(len(self.weights)-1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = self.relu(z)
            self.activations.append(current)
        
        # Output layer with softmax
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        y_pred = self.softmax(z)
        self.activations.append(y_pred)
        
        return y_pred
    
    def backward(self, X, y_true, y_pred):
        """Backward propagation for multi-class classification"""
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient (softmax + cross-entropy)
        delta = y_pred - y_true
        gradients_w[-1] = np.dot(self.activations[-2].T, delta) / m
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers gradients
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
            gradients_w[i] = np.dot(self.activations[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs):
        """Train the neural network using full-batch gradient descent"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Backward pass
            grad_w, grad_b = self.backward(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(grad_w, grad_b)
            
            # Compute loss for monitoring
            loss = self.cross_entropy_loss(y_pred, y)

            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")