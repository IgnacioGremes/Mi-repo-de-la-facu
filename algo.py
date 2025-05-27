import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.array([[4, 1],
              [2, 3],
              [5, 4],
              [1, 0]])

# Centrado
X_centered = X - np.mean(X, axis=0)

# Cálculo de la matriz de covarianza
cov_matrix = (1 / X.shape[0]) * X_centered.T @ X_centered

# Cálculo de autovalores y autovectores
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# Ordenar los autovectores por autovalor decreciente
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# Primer componente principal
pc1 = eigvecs[:, 0]

# Proyecciones
Z = X_centered @ pc1
X_projected = np.outer(Z, pc1) + np.mean(X, axis=0)

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter([4,2,5,1], [1,3,4,0], label="Datos originales", color='blue')
plt.scatter(X_projected[:, 0], X_projected[:, 1], label="Proyecciones", color='orange')
plt.scatter([1/np.sqrt(2), 1/np.sqrt(2) ], [1/np.sqrt(2), - 1/np.sqrt(2) ], label="puntos del eje", color='purple')

# Dibujar el componente principal
origin = np.mean(X, axis=0)
pc1_line = np.array([origin - 3 * pc1, origin + 3 * pc1])
plt.plot(pc1_line[:, 0], pc1_line[:, 1], color='red', label="1er Componente Principal")

# Líneas desde cada punto a su proyección
for i in range(X.shape[0]):
    plt.plot([X[i, 0], X_projected[i, 0]],
             [X[i, 1], X_projected[i, 1]],
             'k--', alpha=0.5)

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("PCA: Proyecciones sobre el primer componente principal")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
