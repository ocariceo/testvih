# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:52:56 2017

@author: ocariceo
"""
#importar los modulos necesarios para la exploracion visual de los datos
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#importar los datos
vih = pd.read_csv("C:/Users/ocariceo/Downloads/vihtest.csv")

plt.figure()
sns.countplot(x='Sexo', hue='Test', data=vih, palette='RdBu')
plt.xticks([0,1], ['Mujer', 'Hombre'])

plt.ylabel("Frecuencia")

plt.show()

#importar el modulo para crear el clasificador

from sklearn.neighbors import KNeighborsClassifier

y = vih['Test'].values
X = vih.drop('Test', axis=1).values

knn = KNeighborsClassifier(n_neighbors = 7)

#ajustar el clasificador
knn.fit(X, y)

y_pred = knn.predict(X)

#simular un nuevo set de datos
X_new = [0,24,1,1]
#aplicar la predicción
new_prediction = knn.predict(X_new)

print("Prediccion: {}".format(new_prediction))

#importar modulos necesarios para ajustar el modelo

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
#Crear los set de datos entrenados y de prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#calcular los valores de precision
print(knn.score(X_test, y_test))


# importar los modulos para establecer la exactitud de los set entrenados y de prueba
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
plt.title('k-NN: Variación del Numero de Vecinos')
plt.plot(neighbors, test_accuracy, label = 'Exactitud del set de prueba')
plt.plot(neighbors, train_accuracy, label = 'Exactitud del set entredado')
plt.legend()
plt.xlabel('Numero of Vecinos')
plt.ylabel('Exactitud')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



