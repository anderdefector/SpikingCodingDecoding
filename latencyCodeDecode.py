import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class latencyCodeDecode:

    def __init__(self, spikes, min_value, max_value, seed):
        self.min_value = min_value
        self.max_value = max_value
        self.spikes = spikes
        self.seed = seed
        self.values_interpolation()

    def code(self, x):
        input_shape = len(x.shape)
        if input_shape == 1:
            x = torch.reshape(x,(x.shape[0],1))
        elif input_shape == 2:
            pass
        samples = x.shape[0]
        features = x.shape[1]
        torch.manual_seed(self.seed)
        feature_spikes = features * self.spikes
        self.coded_Data = torch.zeros(samples, feature_spikes)
        for i in range(samples):
            for j in range(features):
                spike_data = spikegen.rate(x[i, j], num_steps=self.spikes)
                self.coded_Data[i, (j*self.spikes):((j+1)*self.spikes)] = spike_data
        return self.coded_Data

    def values_interpolation(self):
        self.values = np.zeros((self.spikes))
        m = (self.max_value - self.min_value)/( self.spikes )
        for i in range(self.spikes+2):
            if i > 0 and i < self.spikes +1:
                self.values[i - 1] = (m * i) + self.min_value
        return self.values

    def decode(self, s):
        input_shape = len(s.shape)

        if input_shape == 1:
            s = torch.reshape(s,(s.shape[0],1))
        elif input_shape == 2:
            pass
        samples = s.shape[0]
        features = s.shape[1] // self.spikes
        self.decoded_Data = torch.zeros(samples, features)
        for i in range(samples):
            for j in range(features):
                spike_data = s[i, (j*self.spikes):((j+1)*self.spikes)]
                number_spikes = torch.sum(spike_data)
                if int(number_spikes.item()) == 0:
                    self.decoded_Data[i, j] = 0.0
                else:
                    self.decoded_Data[i, j] = self.values[int(number_spikes.item())-1]
        return self.decoded_Data

    def save_decodedData(self,name=None):
        numpy_data = self.decoded_Data.numpy()
        if name == None:
            np.savetxt("RateDecodedData.txt", numpy_data)
        else:
            np.savetxt(name, numpy_data)

    def save_codedData_binary(self, name=None):
        numpy_data = self.coded_Data.numpy()
        if name == None:
            np.savetxt("RateCodedDataBinary.txt", numpy_data, fmt='%d')
        else:
            np.savetxt(name, numpy_data)
    
    def metrics(self, original, decoded):
        mae = mean_absolute_error(original, decoded)
        mse = mean_squared_error(original, decoded)
        r2 = r2_score(original, decoded)
        print("Metrics report : Original vs Decoded")
        print("------------------------------------")
        print("R2: {:.4f}, MAE: {:.4f}, MSE: {:.4f}".format(r2, mae, mse))
        return mae, mse, r2
    
    def plot_signal(self, original, encoded, decoded, signal, name=None):
        input_shape = len(original.shape)
        if input_shape == 1:
            original = torch.reshape(original,(original.shape[0],1))
            decoded = torch.reshape(decoded,(decoded.shape[0],1))
        elif input_shape == 2:
            pass
        data_original = original[:,signal]
        data_encoded = encoded[:,signal*self.spikes:((signal+1)*self.spikes)]
        data_decoded = decoded[:,signal]
        samples = encoded.shape[0]

        fig, axs = plt.subplots(2, figsize=(12.8, 7.2))
        axs[0].plot(data_original,color='red', label='Original')
        axs[0].plot(data_decoded,'--', color='blue', label='Decoded')
        axs[0].legend()
        for i in range(samples):
            for j in range(self.spikes):
                if data_encoded[i,j] == 1:
                    axs[1].scatter(i, j, s=2, color='black')
        axs[1].set_xlabel("Samples")
        axs[1].set_ylabel("Neurons")
        axs[0].set_ylabel("Normalized value")
        if name == None:
            plt.savefig('SignalsRateDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')



