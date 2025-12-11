"""
Modulo PCA implementado con el Metodo de la Potencia

Implementa el analisis de componentes principales (PCA) usando el metodo de la potencia
para encontrar los vectores y valores propios de la matriz de covarianza.

Autores:
  - David Fernandez Vila
  - Alfonso Garrido Bedmar
  - Carlota
  - Pablo Cutillas Bonet
"""

import numpy as np

"""
    Clase PCAMetodoPotencia
    Realiza el Analisis de Componentes Principales (PCA) mediante el Metodo de la potencia.
    
    Este metodo halla los vectores propios de mayor valor propio mediante multiplicaciones 
    matriciales sucesivas.
    
    Atributos:
        n_components (int): Numero de componentes principales a calcular.
        max_iter (int): Numero de iteraciones maximas por vector propio.
        tol (float): Tolerancia de la convergencia.
        mean_ (ndarray): Media de los datos de entrenamiento.
        components_ (ndarray): Vectores propios (componentes principales).
        explained_variance_ (ndarray): Varianza explicada por cada componente.
"""
class PCAMetodoPotencia:
    """
    Costructor de la clase PCA.
        
    Argumentos:
        n_components (int): Numero de componentes principales a extraer.
        max_iter (int): Numero de iteraciones maximas por vector propio.
        tol (float): Tolerancia de la convergencia.
    """
    def __init__(self, n_components, max_iter, tol):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self._covariance_matrix = None
        self._X_centrado = None

    """
    Encuentra el vector propio dominante utilizando el metodo de la potencia.

    El metodo de la potencia multiplica iterativamente un vector por la matriz A,
    normalizando en cada paso. Converge al vector propio con mayor valor propio.

    Argumentos:
        A (ndarray): Matriz de covarianza (cuadrada).
        v_inicial (ndarray): Vector inicial. Si es None, se genera aleatoriamente.
        max_iter (int): Numero de iteraciones maximas por vector propio.
        tol (float): Tolerancia de convergencia.
        
    Devuelve: (vector propio, valor propio)
    """
    def _metodo_potencia(self, A, v_inicial=None, max_iter=1000, tol=1e-6):
        
        n = A.shape[0]
        
        # Inicializamos con un vector aleatorio o el proporcionado
        if v_inicial is None:
            v = np.random.randn(n)
        else:
            v = v_inicial.copy()
        
        # Normalizamos el vector inicial
        v = v / np.linalg.norm(v)
        
        valor_propio_previo = 0

        # Iteraciones del metodo de la potencia
        for _ in range(max_iter):
            v_nuevo = A @ v
            
            # Calculamos el valor propio (cociente de Rayleigh)
            valor_propio = np.dot(v, v_nuevo) / np.dot(v, v)
            
            # Normalizamos
            v_nuevo_normalizado = v_nuevo / np.linalg.norm(v_nuevo)
            
            # Verificamos convergencia (cambio en el vector)
            error = np.linalg.norm(v_nuevo_normalizado - v)
            v = v_nuevo_normalizado
            
            if error < tol:
                break
            if abs(valor_propio - valor_propio_previo) < self.tol * abs(valor_propio):
                break
        valor_propio_previo = valor_propio

        return v, valor_propio
    
    """
    Deflacion:
    Despues de encontrar un vector propio, se elimina de la matriz
    para hallar el siguiente vector propio mas grande.

    Argumentos:
        A (ndarray): Matriz de covarianza.
        vector propio (ndarray): vector propio encontrado.
        valor_propio (float): Valor propio correspondiente.
        
    Devuelve la matriz deflacionada (sin el componente anterior).
    """
    def _deflacion(self, A, v_propio, valor_propio):
        # A_nuevo = A - lambda * u * u^T
        return A - valor_propio * np.outer(v_propio, v_propio)
    
    """
    Ajusta el modelo PCA a los datos de entrenamiento.
        
    Argumentos:
        X (ndarray): Datos de entrenamiento (n_samples, n_features).
        
    Devuelve el objeto para encadenar los metodos.
    """
    def fit(self, X):
        n_samples, n_features = X.shape

        # Calculamos media y centramos datos
        self.mean_ = np.mean(X, axis=0)
        self._X_centrado = X - self.mean_
        
        # Numero de componentes a calcular
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        self.n_components = min(self.n_components, n_features)
        
        # Calculamos matriz de covarianza: Cov = (1/n) * X^T * X
        self._covariance_matrix = (self._X_centrado @ self._X_centrado.T) / (n_samples - 1)

        A = self._covariance_matrix.copy()
        
        # Inicializamos almacenamiento para vectores propios y valores propios
        vectores_propios = []
        valores_propios = []
        
        # Encontramos 'n_components' vectores_propios usando deflacion
        for i in range(self.n_components):
            # Metodo de potencia para encontrar el vector propio dominante
            v, lam = self._metodo_potencia(A, max_iter=self.max_iter, tol=self.tol)
            
            v_original = self._X_centrado.T @ v
            v_original /= np.linalg.norm(v_original)
            
            vectores_propios.append(v_original)
            valores_propios.append(max(lam, 0)) # Aseguramos no negatividad
            
            # Deflacion
            A = self._deflacion(A, v, lam)
        
            # Mostramos el progreso
            if (i + 1) % 10 == 0:
                    print("Extraidos ",i+1, " de ", self.n_components, " componentes.")

        # Almacenamos resultados
        self.components_ = np.array(vectores_propios)
        self.explained_variance_ = np.array(valores_propios)

        return self
    
    """
    Proyecta los datos en el espacio de componentes principales.

    Argumentos:
        X (ndarray): Datos a transformar (n_samples, n_features).

    Devuelve los datos proyectados (n_samples, n_components).
    """
    def transform(self, X):
        
        if self.components_ is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit() primero.")
        
        # Centramos datos
        X_centrado = X - self.mean_
        
        # Proyectamos: X_transformado = X_centrado @ components^T
        return X_centrado @ self.components_.T

    """
    Ajusta el modelo y transforma los datos en un solo paso.

    Argumentos:
        X (ndarray): Datos de entrenamiento y transformacion.
        
    Devuelve los datos proyectados.
    """
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    """
    Reconstruye los datos originales desde los componentes principales.
        
    Argumentos:
        X_transformed (ndarray): Datos proyectados (n_samples, n_components).
        
    Devuelve los datos reconstruidos.
    """
    def inverse_transform(self, X_transformed):
        
        if self.components_ is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit() primero.")
        
        # Reconstruir: X_orig ~= X_transformed @ components + mean_
        return X_transformed @ self.components_ + self.mean_

    """
    Calcula el porcentaje de varianza explicada por cada componente.
        
    Devuelve el ratio de varianza explicada (suma = 1.0).
    """
    def get_variance_ratio(self):
        if self.explained_variance_ is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit() primero.")
        
        total_variance = np.sum(self.explained_variance_)
        if total_variance == 0:
            return np.zeros_like(self.explained_variance_)
        return self.explained_variance_ / total_variance

    """
    Calcula el porcentaje acumulado de varianza explicada.

    Devuelve la varianza acumulada por componente.
    """
    def get_cumulative_variance_ratio(self):
        return np.cumsum(self.get_variance_ratio())
