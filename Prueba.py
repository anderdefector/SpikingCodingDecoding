from CodeDecode import rateCodeDecode, latencyCodeDecode
import numpy as np
import torch

datos_Entrenamiento = np.loadtxt("mgts/Prueba.txt", dtype=np.float32)
X_ent = torch.from_numpy(datos_Entrenamiento[:,0])
Y_ent = torch.from_numpy(datos_Entrenamiento[:,1])

datos_Entrenamiento_torch = torch.from_numpy(datos_Entrenamiento)



code = rateCodeDecode(32, 0.0, 1.0, 42)

codificada = code.code(X_ent)
code.save_codedData_binary()

deco = code.decode(codificada)
code.save_decodedData()
code.plot_signal(X_ent, codificada, deco, 0)

code.metrics(X_ent, deco)

lat = latencyCodeDecode(32, 0.0, 1.0)

codificada = lat.code(datos_Entrenamiento_torch)
lat.save_codedData_binary()

deco = lat.decode(codificada)
lat.save_decodedData()
lat.plot_signal(X_ent, codificada, deco, 0)

lat.metrics(X_ent, deco)