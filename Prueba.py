from rateCodeDecode import rateCodeDecode
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
