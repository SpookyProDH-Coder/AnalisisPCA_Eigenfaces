"""
Programa principal
Autores:
  - David Fernandez Vila
  - Alfonso Garrido Bedmar
  - Carlota
  - Pablo Cutillas Bonet

Codigo original: https://es.python-3.com/?p=4345
"""

from pca_potencia import PCAMetodoPotencia
import numpy as np
import kagglehub
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob
import os

# ---------------- Constantes globales ---------------- #
N_PRUEBAS = 16
N_COMPONENTES = 50
N_MAX_ITERACIONES = 30
N_TOL = 1e-2

# ---------------- Funciones ---------------- #

""" 
    Funcion realizar_query
    Carga una imagen s/nom desde el fichero principal
    y devuelvemos el vector fila (1 x num_pixeles).

    Argumentos:
        (srt) dir - Directorio de la muestra de datos
        (str) s   - La carpeta de la imagen a leer
        (str) nom - El nombre exacto "*.pgm" del fichero a extraer

    Devuelve la query correctamente formateada (1 x num_pixeles)
"""
def realizar_query(dir, s, nom):
    query_path = os.path.join(dir, s, nom)
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    return query_img.reshape(1, -1)

""" 
    Funcion realizar_comparativa
    Representa la diferencia entre la cara que queremos que busque con la cara del modelo. 
    Mostramos el resultado por terminal (la distancia euclidea entre ambos), junto con la comparacion de las caras
    en una grafica de matplotlib.

    Argumentos:
        query       - Dato a comparar de nuestra muestra de datos
        faceshape   - Formato del query
        weights     - El nombre exacto "*.pgm" del fichero a extraer
        eigenfaces  - Las caras propias de nuestra muestra
"""
def realizar_comparativa(query, faceshape, eigenfaces):
    weights = eigenfaces @ (facematrix - pca.mean_).T
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("Mejor match %s con la distancia euclideana %f" % (facelabel[best_match], euclidean_distance[best_match]))

    # Comparativa visual matplotlib
    fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Cara a buscar")
    axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Mejor match")
    plt.show()

# ------------------- #
# Programa principal
# ------------------- #

if __name__ == "__main__":

    print("-"*15, "PROYECTO DE METODOS NUMERICOS", "-"*10)
    print("Obtencion y reconocimiento de caras con el PCA (usando el metodo de la potencia).")

    # ---------------- Descarga y carga de datos ---------------- #
    # Descargamos nuestra base de datos
    print("[*] Descargando la base de datos de las caras...")
    path = kagglehub.dataset_download("kasikrit/att-database-of-faces")
    print("Fichero de datos a leer: ", path)

    # Extraemos las imagenes y las insertamos en un diccionario
    print("[*] Cargando imagenes en memoria...")
    caras = {}
    for persona_dir in sorted(glob(os.path.join(path, "s*"))):
        label = int(os.path.basename(persona_dir).replace("s", ""))

        for img_path in sorted(glob(os.path.join(persona_dir, "*.pgm"))):
            rel = os.path.relpath(img_path, path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue    # No la insertamos
            caras[rel] = {
                "imagen": img,
                "label": label,
            }

    # ---------------- Mostrar muestras ---------------- #
    print("[*] Mostrando ", N_PRUEBAS, "muestras faciales aleatorias...")
    fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
    fig.suptitle("Muestras de caras de la nuestra base de datos.")

    caras_valores = list(caras.values())[-N_PRUEBAS:]
    muestras_caras = [c["imagen"] for c in caras_valores] # Tomamos N_PRUEBAS imagenes

    for i in range(N_PRUEBAS):
        ax = axes[i // 4, i % 4]    # Fila, columna
        ax.imshow(muestras_caras[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()  # Evitamos solapamiento de imagenes
    plt.show()

    # ---------------- Información basica ---------------- #
    print("[*] Analizando datos...")
    primero = list(caras.values())[0]
    faceshape = primero["imagen"].shape
    clases = set(fichero.split("/")[0] for fichero in caras.keys())

    print("\t- Forma de cada imagen (pixeles):", faceshape)
    print("\t- Total de clases (personas):", len(clases))
    print("\t- Numero de imagenes:", len(caras))

    # ---------------- Construimos la matriz de caras ---------------- #
    # Clases 1-39 reservadas para eigenfaces.
    # Mantenemos la clase 40 y la imagen s39/10.pgm como test fuera de muestra

    print("[*] Construimos matrices de caras para el PCA")
    print("\t Calculamos eigenfaces de 1-39, reservamos la clase 40 y 39/10.pgm para tests.")
    
    facematrix = []
    facelabel = []

    for key,val in caras.items():
        carpeta, nombre = key.split("/")
        if carpeta == "s40":
            continue
        elif carpeta == "s39" and nombre == "10.pgm":
            continue
        facematrix.append(val["imagen"].flatten())
        facelabel.append(carpeta)

    # Creamos una matriz NxM con N imagenes y M pixeles por imagenes
    facematrix = np.array(facematrix)

    # ---------------- PCA (eigenfaces) ---------------- #
    print("[*] Aplicamos el PCA con ", N_COMPONENTES, " componentes...")

    # Creamos la instancia de nuestro modelo PCA
    pca = PCAMetodoPotencia(
        n_components=N_COMPONENTES,
        max_iter=N_MAX_ITERACIONES,
        tol=N_TOL
    )

    # Ajustamos el modelo
    pca.fit(facematrix)
    eigenfaces = pca.components_[:N_COMPONENTES]

    print("[*] PCA completado.")
    print("Eigenfaces extraidos: ", eigenfaces.shape[0])

    # Mostramos los primeras N_PRUEBAS caras propias
    print("[*] Mostrando las primeras ", N_PRUEBAS, " caras propias...")
    fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
    for i in range(N_PRUEBAS):
        axes[i // 4][i % 4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
        axes[i // 4, i % 4].axis("off")

    print("Mostrando las caras propias...")
    plt.tight_layout()
    plt.show()

    # ---------------- Proyección (weights) ---------------- #
    # Generamos pesos como una matriz KxN, con K el numero de eigenfaces y N el numbero de muestras

    weights = eigenfaces @ (facematrix - pca.mean_).T
    print("Forma de la matriz de pesos:", weights.shape)

    # Mostramos la varianza
    var_ratio = pca.get_variance_ratio()
    cum_var_ratio = pca.get_cumulative_variance_ratio()

    print("[*] Analisis realizado:")
    print("Varianzas obtenidas de los primeros 10 componentes:")
    for i in range(min(10, len(var_ratio))):
        print("[", i+1, "]: ", var_ratio[i]*100)

    # ------ Tests con imagenes fuera de nuestro modelo ------ #
    # Buscamos la s39/10.pgm y la s40/1.pgm, que se encuentran fuera de nuestro modelo,
    # y las comparamos con las caras encontradas en nuestro modelo.

    print("[*] Realizamos el test comparando imagenes fuera del modelo...")
    print("- Test 1: Cara s39/10.pgm:")
    query = realizar_query(path, "s39", "10.pgm")
    realizar_comparativa(query, faceshape, eigenfaces)

    print("- Test 2: Cara s40/1.pgm:")
    query = realizar_query(path, "s40", "1.pgm")
    realizar_comparativa(query, faceshape, eigenfaces)