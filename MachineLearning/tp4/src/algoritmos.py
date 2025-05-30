import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from collections import deque
from scipy.spatial.distance import cdist

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def k_means(X, cant_clust, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    random_indices = random.sample(range(n_samples), cant_clust)
    centroids = X[random_indices]

    for _ in range(max_iter):
        # E-step
        clusters = [[] for _ in range(cant_clust)] 
        for sample in X:
            distance = [euclidean_distance(sample, centroid) for centroid in centroids]
            assigned_cluster = np.argmin(distance)
            clusters[assigned_cluster].append(sample)

        # M-step
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                          for i, cluster in enumerate(clusters)])

        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

        final_assigned_clust = []
        for sample in X:
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            final_assigned_clust.append(np.argmin(distances))
        
    return np.array(final_assigned_clust), centroids   

def graf_ganancias_decrecientes(X, K_max):
    wcss = []  # Suma de errores cuadráticos intra-clúster
    for k in range(1, K_max + 1):
        labels, centroids = k_means(X, k)
        sum_squared_error = 0
        for i, sample in enumerate(X):
            centroide = centroids[labels[i]]
            sum_squared_error += euclidean_distance(sample, centroide)**2
        wcss.append(sum_squared_error)

    # Graficar el codo
    plt.plot(range(1, K_max + 1), wcss, marker='o')
    # plt.title("Método del Codo")
    plt.xlabel("Número de Clústeres (K)")
    plt.ylabel("Suma de errores cuadráticos (WCSS)")
    plt.grid(True)
    plt.show()

def graf_clusters(X, K):
    labels, centroids = k_means(X, K)
    
    # Choose colormap based on K
    cmap = cm.get_cmap('tab20', K) if K <= 20 else cm.get_cmap('nipy_spectral', K)

    plt.figure(figsize=(8, 6))

    # Plot each cluster with a distinct color
    for k in range(K):
        puntos = X[labels == k]
        plt.scatter(puntos[:, 0], puntos[:, 1], color=cmap(k), label=f'Clúster {k}', s=40)

    # Plot centroids in black
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroides')

    # plt.title('Visualización de Clústeres')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.legend()
    plt.grid(True)
    plt.show()

def calcular_responsabilidades(X, pi, mu, Sigma):
    n_samples = X.shape[0]
    K = len(pi)
    gamma = np.zeros((n_samples, K))  # responsabilidad γ_ik

    for k in range(K):
        # Crear objeto distribución gaussiana multivariada
        dist = multivariate_normal(mean=mu[k], cov=Sigma[k])
        # Calcular f(x_i | mu_k, Sigma_k) para todos los puntos
        gamma[:, k] = pi[k] * dist.pdf(X)  # vectorizado

    # Normalizar para que sumen 1 por fila
    gamma /= gamma.sum(axis=1, keepdims=True)

    return gamma  # shape: (n_samples, K)

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def inicializar_parametros(self, X):
        n_samples, n_features = X.shape

        # Get labels and centroids from K-means
        labels, centroids = k_means(X, self.K)
        self.mu = centroids
        self.pi = np.zeros(self.K)
        self.Sigma = np.zeros((self.K, n_features, n_features))

        for k in range(self.K):
            X_k = X[labels == k]
            self.pi[k] = len(X_k) / n_samples
            if len(X_k) > 1:
                self.Sigma[k] = np.cov(X_k.T) + 1e-6 * np.eye(n_features)
            else:
                # In case a cluster has only one point, use global covariance as fallback
                self.Sigma[k] = np.cov(X.T) + 1e-6 * np.eye(n_features)

    def calcular_responsabilidades(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.K))

        for k in range(self.K):
            dist = multivariate_normal(mean=self.mu[k], cov=self.Sigma[k], allow_singular=True)
            gamma[:, k] = self.pi[k] * dist.pdf(X)

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def actualizar_parametros(self, X, gamma):
        n_samples, n_features = X.shape
        Nk = gamma.sum(axis=0)  # tamaño efectivo de cada componente

        self.pi = Nk / n_samples
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]
        self.Sigma = np.zeros((self.K, n_features, n_features))

        for k in range(self.K):
            diff = X - self.mu[k]
            weighted_sum = np.einsum("ni,nj->ij", diff * gamma[:, k, np.newaxis], diff)
            self.Sigma[k] = weighted_sum / Nk[k] + 1e-6 * np.eye(n_features)

    def log_verosimilitud(self, X):
        n_samples = X.shape[0]
        log_likelihood = np.zeros(n_samples)

        for k in range(self.K):
            dist = multivariate_normal(mean=self.mu[k], cov=self.Sigma[k], allow_singular=True)
            log_likelihood += self.pi[k] * dist.pdf(X)

        return np.sum(np.log(log_likelihood))

    def fit(self, X):
        self.inicializar_parametros(X)

        log_likelihood_old = None
        for i in range(self.max_iter):
            # E-step
            gamma = self.calcular_responsabilidades(X)
            # M-step
            self.actualizar_parametros(X, gamma)
            log_likelihood = self.log_verosimilitud(X)

            if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

    def predict(self, X):
        gamma = self.calcular_responsabilidades(X)
        return np.argmax(gamma, axis=1)
    
def graf_ganancias_decrecientes_gmm_distancia(X, K_max):
    distancias = []

    for k in range(1, K_max + 1):
        gmm = GMM(n_components=k)
        gmm.fit(X)
        responsabilidades = gmm.calcular_responsabilidades(X)
        asignaciones = np.argmax(responsabilidades, axis=1)

        suma_distancias = 0
        for i, x in enumerate(X):
            mu_k = gmm.mu[asignaciones[i]]
            suma_distancias += np.linalg.norm(x - mu_k) ** 2

        distancias.append(suma_distancias)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, K_max + 1), distancias, marker='o')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Suma de Distancias al Centro (tipo WCSS)')
    # plt.title('Ganancias Decrecientes usando GMM (Distancia a la Media)')
    plt.grid(True)
    plt.show()

def plot_neg_log_likelihood_vs_k(X, k_values):
    neg_log_likelihoods = []

    for k in range(1, k_values + 1):
        gmm = GMM(n_components=k)
        gmm.fit(X)
        ll = gmm.log_verosimilitud(X)
        neg_log_likelihoods.append(-ll)  # usamos negativo para visualizar

    # Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k_values + 1), neg_log_likelihoods, marker='o')
    plt.xlabel("Número de componentes (K)")
    plt.ylabel("- Log-verosimilitud")
    # plt.title("Evaluación de GMM: -loglikelihood vs. K")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cluster_GMM(X, gmm):
    gamma = gmm.calcular_responsabilidades(X)
    asignaciones = np.argmax(gamma, axis=1)
    K = gmm.K
    
    cmap = cm.get_cmap('tab20', K) if K <= 20 else cm.get_cmap('nipy_spectral', K)

    plt.figure(figsize=(8, 6))
    for k in range(K):
        puntos_k = X[asignaciones == k]
        plt.scatter(puntos_k[:, 0], puntos_k[:, 1], color=cmap(k), label=f'Cluster {k}', s=40)

    # Plot centroids in black with 'x' marker
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c='black', marker='x', s=100, label='Centroides')

    # plt.title('Clusters GMM y sus Centroides')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.legend()
    plt.grid(True)
    plt.show()

def dbscan(X, eps=0.5, min_samples=5):
    n_samples = len(X)
    labels = np.full(n_samples, -1)  # -1 representa "ruido"
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    def region_query(i):
        # Devuelve índices de puntos dentro del radio eps del punto i
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances <= eps)[0]

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        vecinos = region_query(i)
        if len(vecinos) < min_samples:
            labels[i] = -1  # ruido
        else:
            labels[i] = cluster_id
            cola = deque(vecinos)
            while cola:
                j = cola.popleft()
                if not visited[j]:
                    visited[j] = True
                    vecinos_j = region_query(j)
                    if len(vecinos_j) >= min_samples:
                        cola.extend(vecinos_j)
                if labels[j] == -1:
                    labels[j] = cluster_id  # anteriormente ruido, ahora parte del clúster
                if labels[j] == -1 or labels[j] == -2:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels


def plot_cluster_DBSCAN(X, labels):
    unique_labels = set(labels)
    K = len(unique_labels - {-1})  # number of clusters (excluding noise)

    colors = cm.get_cmap('tab20', K)  # enough unique colors

    plt.figure(figsize=(8, 6))
    
    for k in unique_labels:
        class_members = labels == k
        if k == -1:
            # Noise points in black
            plt.scatter(X[class_members, 0], X[class_members, 1], c='black', label='Noise', s=40)
        else:
            # Cluster points with distinct color
            plt.scatter(X[class_members, 0], X[class_members, 1], 
                        color=colors(k), label=f'Cluster {k}', s=40)
    
    # plt.title('DBSCAN Clustering')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.legend()
    plt.grid(True)
    plt.show()

class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.S = None  # Store singular values (optional)

    def _ensure_numpy(self, X):
        return np.asarray(X)

    def fit(self, X):
        X = self._ensure_numpy(X)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        self.S = S  # Save for optional use (e.g., plotting)

    def transform(self, X):
        X = self._ensure_numpy(X)
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean

    def reconstruction_error(self, X):
        X = self._ensure_numpy(X)
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        return np.mean((X - X_reconstructed) ** 2)

    def plot_singular_values(self):
        if self.S is None:
            raise RuntimeError("You must call fit() before plotting singular values.")
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(self.S) + 1), self.S)
        plt.xlabel('Index of Singular Value')
        plt.ylabel('Singular Value')
        # plt.title('Singular Values from SVD')
        plt.grid(True)
        plt.show()

def train_val_split(array, val_ratio=0.2):
    # np.random.seed(seed)

    n_samples = len(array)
    indices = np.random.permutation(n_samples)

    split_point = int(n_samples * (1 - val_ratio))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_array = array[train_indices]
    val_array = array[val_indices]

    return train_array, val_array