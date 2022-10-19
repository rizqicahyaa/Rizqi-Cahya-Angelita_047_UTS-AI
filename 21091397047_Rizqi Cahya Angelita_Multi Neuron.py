# NAMA : RIZQI CAHYA ANGELITA
# NIM : 21091397047
# KELAS : 2021A

#Multi neuron dengan memakai NumPy

#Mengimpor library NumPy
import numpy as np

#Menginisialisasi variabel dengan jumlah 10 input
inputs = [4.0, 2.0, 9.0, 7.0, 1.0, 3.0, 2.7, 5.0, 8.0, 6.0]

#Menginisialisasi bobot neuron dan bias
weights = [[0.8, 0.6, 0.5, 0.2, 0.13, 0.9, 0.3, 0.22, 0.5, -0.7],
           [0.17, 0.21, 0.25, 0.10, 0.14, 0.16, 0.27, 0.11, 0.19, 0.12],
           [0.29, 0.30, 0.31, 0.18, 0.15, -0.12, -0.45, 0.32, 0.24, -0.7],
           [1.0, 1.8, 1.5, 1.6, 1.7, 1.3, 1.4, 1.1, 1.9, 1.12],
           [2.0, 1.2, 0.1, 7.2, -0.23, -0.17, 6.0, -0.7, 0.29, -0.64]]

biases = [2.0, 6.0, 5.0, 8.0, 3.0]

#Menghitung output menggunakan fungsi dot produk, np.dot()
layer_outputs = np.dot(weights, inputs) + biases

#Mencetak output
print(layer_outputs)