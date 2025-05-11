import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

class NeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.1, num_classes=49, schedule=None, gamma_decay=0.95, use_mini_batch=False, batch_size=32, optimizer='gd', l2_lambda=0.0):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.num_classes = num_classes
        self.schedule = schedule
        self.gamma_decay = gamma_decay
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
            self.learning_rate = max(self.initial_learning_rate * (1 - epoch / epochs), 0.0001)
        elif self.schedule == "exponential":
            self.learning_rate = self.initial_learning_rate * (self.gamma_decay ** epoch)

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
    
    def get_best_val_stats(self, X_val, Y_val, val_losses):
        if not val_losses:
            raise ValueError("No validation losses recorded.")
        
        best_epoch = int(np.argmin(val_losses))
        best_val_loss = val_losses[best_epoch]

        Y_pred_probs, _ = self.forward(X_val)
        val_accuracy = self.compute_accuracy(Y_pred_probs, Y_val)

        return best_val_loss, val_accuracy, best_epoch