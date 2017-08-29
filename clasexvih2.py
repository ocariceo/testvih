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
exvih = pd.read_csv("C:/Users/ocariceo/Downloads/exvih.csv")

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


# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('Valor de Falsos Positivos')
plt.ylabel('Valor de Verdaderos Positivos')
plt.title('Curva ROC')
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv = 5, scoring = "roc_auc")

# Print list of AUC scores
print("Valor AUC calculado usando 5 etapas de validación cruzada: {}".format(cv_auc))










