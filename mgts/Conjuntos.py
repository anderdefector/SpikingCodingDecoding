import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


Datos = np.loadtxt("mgdata.txt")

print(Datos.shape)

lon_entrenamiento = math.floor(Datos.shape[0]*0.70)

lon_validacion = math.floor(Datos.shape[0]*0.20)

lon_prueba = math.floor(Datos.shape[0]*0.10)

scaler.fit(Datos)
datos_esc = scaler.transform(Datos)

print(str(lon_entrenamiento) + " " + str(lon_validacion) + " "+ str(lon_prueba))

Entrena = np.zeros((lon_entrenamiento, 2)) 
Validacion = np.zeros((lon_validacion, 2))
Prueba = np.zeros((lon_prueba, 2))

Entrena[:, 0] = datos_esc[0:lon_entrenamiento, 1]
Entrena[:, 1] = datos_esc[1:lon_entrenamiento+1, 1]

np.savetxt("Entrenamiento.txt", Entrena)

a = lon_entrenamiento+1

b = lon_entrenamiento+lon_validacion+1
print(a)
print(b)

Validacion[:, 0] = datos_esc[lon_entrenamiento:lon_entrenamiento+lon_validacion, 1]
Validacion[:, 1] = datos_esc[lon_entrenamiento+1:lon_entrenamiento+lon_validacion+1, 1]

np.savetxt("Validacion.txt", Validacion)

Prueba[:, 0] = datos_esc[lon_entrenamiento+lon_validacion:lon_entrenamiento+lon_validacion+lon_prueba, 1]
Prueba[:, 1] = datos_esc[lon_entrenamiento+lon_validacion+1:lon_entrenamiento+lon_validacion+lon_prueba+1, 1]



np.savetxt("Prueba.txt", Prueba)
