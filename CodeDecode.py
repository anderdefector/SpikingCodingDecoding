import numpy as np
import torch
from snntorch import spikegen
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class rateCodeDecode:

    def __init__(self, spikes, min_value, max_value, seed):
        self.min_value = min_value
        self.max_value = max_value
        self.spikes = spikes
        self.seed = seed
        self.values_interpolation()

    def code_snntorch(self, x):
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
    def code(self, x):
        input_shape = len(x.shape)
        if input_shape == 1:
            x = torch.reshape(x,(x.shape[0],1))
        elif input_shape == 2:
            pass
        samples = x.shape[0]
        features = x.shape[1]
        feature_spikes = features * self.spikes
        self.coded_Data = torch.ones(samples, feature_spikes)
        for i in range(samples):
            for j in range(features):
                spike_position = int(round(x[i, j].item() / self.m, 0 ))
                if spike_position == 0:
                    t = torch.zeros(self.spikes)
                    self.coded_Data[i, (j*self.spikes):((j+1)*self.spikes)] = t
                else:
                    t = torch.zeros(self.spikes-spike_position)
                    tmp = self.coded_Data[i, (j*self.spikes):((j+1)*self.spikes)]
                    tmp[spike_position:self.spikes] = t
                    self.coded_Data[i, (j*self.spikes):((j+1)*self.spikes)] = tmp
        return self.coded_Data


    def values_interpolation(self):
        self.values = np.zeros((self.spikes))
        self.m = (self.max_value - self.min_value)/( self.spikes )
        for i in range(self.spikes+2):
            if i > 0 and i < self.spikes +1:
                self.values[i - 1] = (self.m * i) + self.min_value
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
            np.savetxt(name, numpy_data, fmt='%d')
    
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
        axs[0].legend(fontsize=12)
        first_spike = True
        for i in range(samples):
            for j in range(self.spikes):
                if data_encoded[i,j] == 1:
                    label = 'Encoded' if first_spike else None
                    axs[1].scatter(i, j, s=2, color='black', label=label)
                    first_spike = False
        axs[1].set_xlabel("Samples", fontsize=14)
        axs[1].set_ylabel("LIF Neurons", fontsize=14)
        axs[1].set_ylim(0,self.spikes-1)
        axs[0].set_ylabel("Normalized value", fontsize=14)
        axs[0].set_ylim(self.min_value,self.max_value)
        axs[1].legend(fontsize=12)
        if name == None:
            plt.savefig('SignalsRateDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')
        plt.close()

    def plot_signal_optimized(self, original, encoded, decoded, signal, name=None):
        if len(original.shape) == 1:
            original = original.unsqueeze(1)
            decoded = decoded.unsqueeze(1)

        data_original = original[:, signal]
        data_encoded = encoded[:, signal * self.spikes:(signal + 1) * self.spikes]
        data_decoded = decoded[:, signal]

        samples = encoded.shape[0]

        fig, axs = plt.subplots(2, figsize=(12.8, 7.2))

        # Original signal
        axs[0].plot(data_original, color='red', label='Original')
            
        # Decoded signal
        axs[0].plot(data_decoded, '--', color='blue', label='Decoded')
            
        axs[0].legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")

        # Encoded LIF neurons
        first_spike = True
        for i in range(samples):
            for j in range(self.spikes):
                if data_encoded[i, j] == 1:
                    label = 'Encoded' if first_spike else None
                    axs[1].scatter(i, j, s=2, color='black', label=label)
                    first_spike = False

        axs[1].set_xlabel("Samples", fontsize=14)
        axs[1].set_ylabel("LIF Neurons", fontsize=14)
        axs[1].set_ylim(-0.5, ((self.spikes-1) + 0.5))

        axs[0].set_ylabel("Normalized value", fontsize=14)
        axs[0].set_ylim(self.min_value, self.max_value)

        axs[1].legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")

        if name is None:
            plt.savefig('SignalsRateDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')

        plt.close()

class latencyCodeDecode:

    def __init__(self, spikes, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.spikes = spikes
        self.values_interpolation()

    def code(self, x):
        input_shape = len(x.shape)
        if input_shape == 1:
            x = torch.reshape(x,(x.shape[0],1))
        elif input_shape == 2:
            pass
        samples = x.shape[0]
        features = x.shape[1]
        feature_spikes = features * self.spikes
        self.coded_Data = torch.zeros(samples, feature_spikes)
        for i in range(samples):
            for j in range(features):
                spike_data = torch.zeros(self.spikes)
                spike_position = int(round(x[i, j].item() / self.m, 0 ))
                if spike_position == 0:
                    t = self.spikes - 1
                else:
                    t = self.spikes - spike_position
                spike_data[t] = 1.0
                self.coded_Data[i, (j*self.spikes):((j+1)*self.spikes)] = spike_data
        return self.coded_Data

    def values_interpolation(self):
        self.values = np.zeros((self.spikes))
        self.m = (self.max_value - self.min_value)/( self.spikes )
        for i in range(self.spikes+2):
            if i > 0 and i < self.spikes +1:
                self.values[i - 1] = (self.m * i) + self.min_value
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
                sumatory = 0.0
                #print("Spike data")
                #print(spike_data)
                for k in range(self.spikes):
                    if spike_data[k] == 1.0:
                        #print("Pulsos")
                        t = self.spikes - k
                        sumatory = sumatory + ( t * self.m * spike_data[k])
                        
                
                if sumatory == 0.0:
                    self.decoded_Data[i, j] = 0* spike_data[0]
                else:
                    self.decoded_Data[i, j] = sumatory
        return self.decoded_Data

    def decode_firstSpike(self, s):
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
                sumatory = 0.0
                #print("Spike data")
                #print(spike_data)
                spike = False
                for k in range(self.spikes):
                    if spike_data[k] == 1.0:
                        #print("Pulsos")
                        t = self.spikes - k
                        spike = True
                        self.decoded_Data[i, j] = t * self.m * spike_data[k]
                        break
                if spike == False:
                    self.decoded_Data[i, j] = 0 * spike_data[0]
                
        return self.decoded_Data

    def save_decodedData(self,name=None):
        numpy_data = self.decoded_Data.numpy()
        if name == None:
            np.savetxt("LatencyDecodedData.txt", numpy_data)
        else:
            np.savetxt(name, numpy_data)

    def save_codedData_binary(self, name=None):
        numpy_data = self.coded_Data.numpy()
        if name == None:
            np.savetxt("LatencyCodedDataBinary.txt", numpy_data, fmt='%d')
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
        axs[0].legend(fontsize=12)
        first_spike = True
        for i in range(samples):
            for j in range(self.spikes):
                if data_encoded[i,j] == 1:
                    label = 'Encoded' if first_spike else None
                    axs[1].scatter(i, j, s=2, color='black', label=label)
                    first_spike = False
                    #print(label)
                    break
        axs[1].set_xlabel("Samples", fontsize=14)
        axs[1].set_ylabel("LIF Neurons", fontsize=14)
        axs[1].set_ylim(-0.5, (self.spikes + 0.5))
        axs[1].legend(fontsize=12)
        axs[0].set_ylim(self.min_value,self.max_value)
        axs[0].set_ylabel("Normalized value", fontsize=14)
        if name == None:
            plt.savefig('SignalsLatencyDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')
        plt.close()


    def plot_signal_optimized(self, original, encoded, decoded, signal, name=None):
        if len(original.shape) == 1:
            original = original.unsqueeze(1)
            decoded = decoded.unsqueeze(1)

        data_original = original[:, signal]
        data_encoded = encoded[:, signal * self.spikes:(signal + 1) * self.spikes]
        data_decoded = decoded[:, signal]

        samples = encoded.shape[0]

        fig, axs = plt.subplots(2, figsize=(12.8, 7.2))

        # Original signal
        axs[0].plot(data_original, color='red', label='Original')
        
        # Decoded signal
        axs[0].plot(data_decoded, '--', color='blue', label='Decoded')
        
        axs[0].legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")

        # Encoded LIF neurons
        first_spike = True
        for i in range(samples):
            spike_indices = torch.where(data_encoded[i] == 1)[0]
            if spike_indices.size(0) > 0:
                j = spike_indices[0]
                label = 'Encoded' if first_spike else None
                axs[1].scatter(i, j, s=2, color='black', label=label)
                first_spike = False

        axs[1].set_xlabel("Samples", fontsize=14)
        axs[1].set_ylabel("LIF Neurons", fontsize=14)
        axs[1].set_ylim(-0.5, ((self.spikes - 1) + 0.5))
        axs[1].legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")
        axs[0].set_ylabel("Normalized value", fontsize=14)
        axs[0].set_ylim(self.min_value, self.max_value)
        #axs[1].set_ylim(-0.5, self.max_valu)
        
        if name is None:
            plt.savefig('SignalsLatencyDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')

        plt.close()

    def plot_signal_spikescount(self, original, encoded, decoded, signal, name=None):
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

        fig, axs = plt.subplots(3, figsize=(12.8, 7.2))
        axs[0].plot(data_original,color='red', label='Original')
        axs[0].plot(data_decoded,'--', color='blue', label='Decoded')
        axs[0].legend(fontsize=14,  loc='outside right upper')
        for i in range(samples):
            number = torch.sum(data_encoded[i,:])
            axs[2].scatter(i, number, s=1, color='black')
            for j in range(self.spikes):
                if data_encoded[i,j] == 1:
                    axs[1].scatter(i, j, s=2, color='black')
                    break

        
        axs[2].set_ylabel("Number of spikes", fontsize=14)
        axs[2].set_xlabel("Samples", fontsize=14)
        axs[1].set_ylabel("LIF Neurons", fontsize=14)
        axs[0].set_ylabel("Normalized value", fontsize=14)
        if name == None:
            plt.savefig('SignalsLatencyDecoding.png', bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')
        plt.close()