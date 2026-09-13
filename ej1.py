import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Abro las imágenes

ruta_dataset = 'dataset_tp1'

ruta_train = ruta_dataset + '/train'
ruta_test = ruta_dataset + '/test'

ruta_train_labels = ruta_dataset + '/train_labels.csv'
ruta_test_labels = ruta_dataset + '/test_labels.csv'

train_labels = pd.read_csv(ruta_train_labels)
test_labels = pd.read_csv(ruta_test_labels)

X_train = []
y_train = []
X_test = []
y_test = []
i=0

for _, fila in train_labels.iterrows():
    nombre = fila['archivo']
    clase = fila['clase']

    ruta = os.path.join(ruta_train, nombre)

    img = Image.open(ruta).convert('L')
    X_train.append(np.array(img))
    y_train.append(clase)

for _, fila in test_labels.iterrows():
    nombre = fila['archivo']
    clase = fila['clase']

    ruta = os.path.join(ruta_test, nombre)

    img = Image.open(ruta).convert('L')
    X_test.append(np.array(img))
    y_test.append(clase)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


#Hago que cada imagen sea un vector
X_train_vec = X_train.reshape(X_train.shape[0], -1)
X_test_vec = X_test.reshape(X_test.shape[0], -1)


#CASO FOTOS NORMALES
#Creo el modelo y lo entreno con el vector de las imágenes train
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train_vec, y_train)

#Testeo
y_testeado = lr_model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_testeado)

print("Accuracy:", acc)


#CASOS PCA MEDIANTE SVD CON DISTINTOS K
#Resto esperanza a X_train
media_train = np.mean(X_train_vec, axis=0)
X_train_centrado = X_train_vec - media_train
media_test = np.mean(X_test_vec, axis=0)
X_test_centrado = X_test_vec - media_test

#Aplico SVD
U, S, Vt = np.linalg.svd(X_train_centrado, full_matrices=False)

def proyectar_componentes_principales(k):
    Vk=Vt[:k].T
    proyeccion=X_test_centrado@Vk
    return proyeccion
proyeccion_k_2=proyectar_componentes_principales(2)

plt.figure()
sanos=y_test==0
neumonia=y_test==1

plt.grid()
plt.scatter(proyeccion_k_2[sanos,0],proyeccion_k_2[sanos,1],color="blue",label="Sanos")
plt.scatter(proyeccion_k_2[neumonia,0],proyeccion_k_2[neumonia,1],color="red",label="Neumonia")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.show()

#Comienzo a iterar sobre los distintos valores posibles de k
accuracies = []
valores_k = range(2, 200)

for K in valores_k:
    Vk = Vt[:K].T   #Tomar las primeras K componentes

    Z_train = X_train_centrado @ Vk     # Proyectar train
    Z_test = X_test_centrado @ Vk       # Proyectar test

    lr_model = LogisticRegression(max_iter=2000)    #Entrena modelo
    lr_model.fit(Z_train, y_train)

    y_testeado_k = lr_model.predict(Z_test)     #Testeo
    acc = accuracy_score(y_test, y_testeado_k)      #Accuracy
    accuracies.append(acc)
    #print(K)

#En la parte del gráfico hubo "ayuda", no sabía bien cómo hacerlo
plt.figure()
plt.grid()
plt.plot(valores_k, accuracies, label="PCA mediante SVD")

plt.axhline(
    acc,
    linestyle="--",
    label="Entrenado sin PCA"
)

plt.xlabel("Cantidad de componentes K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()