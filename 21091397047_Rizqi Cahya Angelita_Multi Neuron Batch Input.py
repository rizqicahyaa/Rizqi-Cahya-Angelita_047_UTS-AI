# NAMA : RIZQI CAHYA ANGELITA
# NIM : 21091397047
# KELAS : 2021A

#Multi Neuron Batch Input

#Mengimpor library NumPy
import numpy as np

#Menginisialisasi variabel dengan input 10 dan batch 6
inputs = [[0.9, 2.1, 3.2, 1.17, 0.10, 3.7, 4.4, 0.6, 0.5, 5.1],
          [1.3, 1.5, 0.7, 0.8, 2.4, 5.7, 4.2, 2.2, 2.9, 2.9],
          [0.1, 0.5, 1.7, 7.2, 2.1, 6.1, 2.3, 0.4, 1.1, 1.8],
          [3.1, 5.2, 2.9, 1.6, 4.7, 1.9, 7.7, 4.0, 7.8, 5.5],
          [2.0, 2.7, 3.4, 5.0, 1.0, 3.5, 1.4, 0.3, 6.0, 1.21],
          [0.2, 2.5, 4.3, 0.4, 2.8, 7.6, 3.3, 3.0, 6.4, 8.0]]

#Menginisialisasi bobot neuron dan bias
weights = [[3.0, 2.2, 3.5, 2.0, 5.6, 2.6, 2.3, 6.2, 1.3, 1.5],
           [5.1, 0.9, 1.4, 5.4, 7.0, 3.2, 2.1, 2.16, 5.2, 2.10],
           [7.9, 3.5, 7.5, 0.5, 1.9, 1.12, 2.5, 7.3, 2.6, 4.0],
           [3.1, 6.0, 1.2, 1.8, 2.4, 3.3, 2.4, 4.1, 8.2, 1.19],
           [1.6, 1.0, 2.8, 1.15, 0.3, 1.11, 2.7, 3.8, 1.5, 3.4]]

biases = [2.5, 5.1, 2.7, 0.4, 1.0]

#Menghitung output menggunakan fungsi dot produk, np.dot() dan np.array
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

#Mencetak output
print(layer_outputs)