from ej1 import X_train_centrado, Vt, y_train, y_test, X_test_centrado, X_test_vec, media_train
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

V2=Vt[:2].T
Z_train = X_train_centrado @ V2     # Proyectar train
Z_test = X_test_centrado @ V2       # Proyectar test
lr_model = LogisticRegression(max_iter=2000)    #Entrena modelo
lr_model.fit(Z_train, y_train)
y_testeado_2 = lr_model.predict(Z_test)     #Testeo
a0 = accuracy_score(y_test, y_testeado_2) 
def evaluacion_imagenes_rotadas(p):
    x_test_rotado=X_test_vec.copy()

    for i  in range(len(X_test_vec)):
        if(random.random()<=p):
            x_test_rotado[i]=x_test_rotado[i][::-1]
    Z_test_rotado=(x_test_rotado-media_train)@V2
    y_rotado_testeado=lr_model.predict(Z_test_rotado)
    ap=accuracy_score(y_test,y_rotado_testeado)
    return ap

#quizas pueda sacar despues algunos p, los puse más que nada para que quede menos tosco los ultimos dos gráficos
#Despues podria hacerlo en una funcion y generalizar ( en vez de 100 un numero ajustable y poder ajustar la tolerancia, etc)
p_evaluar=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
esperanza_acc=[]
prob_superar_tol=[]
for p in p_evaluar:
    accuracy_img_rotadas=[]
    diferencias_a0_ap=[]
    for i in range(1000):
        ap=evaluacion_imagenes_rotadas(p)
        accuracy_img_rotadas.append(ap)
        diferencias_a0_ap.append(a0-ap)
    plt.hist(accuracy_img_rotadas)
    plt.show()
    esperanza_acc.append(np.mean(accuracy_img_rotadas))
    prob_superar_tol.append(np.sum(np.array(diferencias_a0_ap)>0.1)/1000)

plt.figure()
plt.title("E(Ap)")
plt.grid()
plt.plot(p_evaluar, esperanza_acc, marker="o",label=" Esperanza de los accuracy")
plt.xlabel("Probabilidad de rotar")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.title("P(L(p)>δ)")
plt.grid()
plt.plot(p_evaluar,prob_superar_tol, marker="o",label="Probabilidad de superar la tolerancia")
plt.xlabel("Probabilidad de rotar")
plt.ylabel("Probabilidad de superar la tolerancia")
plt.legend()
plt.show()

