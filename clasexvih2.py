# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:57:44 2017

@author: ocariceo
"""

#importar los modulos necesarios para la exploracion visual de los datos
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#importar los datos
exvih = pd.read_csv("exvih.csv")

plt.figure()
sns.countplot(x='Sexo', hue='Test', data=exvih, palette='RdBu')
plt.xticks([0,1], ['Mujer', 'Hombre'])

plt.ylabel("Frecuencia")

plt.show()

#importar el modulo para crear el clasificador

from sklearn.neighbors import KNeighborsClassifier

y = exvih['Test'].values
X = exvih.drop('Test', axis=1).values

knn = KNeighborsClassifier(n_neighbors = 7)

#ajustar el clasificador
knn.fit(X, y)

y_pred = knn.predict(X)

#simular un nuevo set de datos
X_new = [0,24,1,1]
#aplicar la predicción
new_prediction = knn.predict(X_new)

print("Prediccion: {}".format(new_prediction))

#importa modulos necesarios para ajustar el modelo

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
#Crear los set de datos entrenados y de prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#calcular los valores de precision
print(knn.score(X_test, y_test))


# importar los modulos para establecer la precision de los set entrenados y de prueba
import numpy as np
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# probar la función con diversos valores para el modelo
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)

# generar un grafico para determinar el numero correcto para el clasificador 
plt.title('k-NN: Variación de numeros de Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Precisión del Set de Prueba')
plt.plot(neighbors, train_accuracy, label = 'Precisión del Set Entrenado')
plt.legend()
plt.xlabel('Numero de Neighbors')
plt.ylabel('Precision')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Crear el clasificador
logreg = LogisticRegression()

# ajustar el clasificador a los datos entrenados
logreg.fit(X_train, y_train)

# Predecir las categorias del set de datos de prueba
y_pred = logreg.predict(X_test)

# Calcular la matriz de confusion y el reporte de clasificacion
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Importar modulos necesarios
from sklearn.metrics import roc_curve

# calcular las probabilidades para la prediccion 
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generar el grafico de la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('Valor de Falsos Positivos')
plt.ylabel('Valor de Verdaderos Positivos')
plt.title('Curva ROC')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Calcular los valores del area bajo la curva (AUC) 
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Calcular los valores de la validacion cruzada de AUC
cv_auc = cross_val_score(logreg, X, y, cv = 5, scoring = "roc_auc")

print("Valor AUC calculado usando 5 etapas de validación cruzada: {}".format(cv_auc))










