import numpy as np
import pandas as pd
from PIL import Image
import os

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

