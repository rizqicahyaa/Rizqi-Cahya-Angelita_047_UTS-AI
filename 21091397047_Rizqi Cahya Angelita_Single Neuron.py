# NAMA : RIZQI CAHYA ANGELITA
# NIM : 21091397047
# KELAS : 2021A

#Single Neuron dengan memakai NumPy

#Mengimpor library NumPy
import numpy as np

#Menginisialisasi variabel dengan jumlah 10 input
inputs = [7, 5, 1, 3, 2, 9, 6, 8, 4, 3.5]

#Menginisialisasi bobot neuron dan bias
weights = [0.1, 0.5, 0.3, 1.3, 0.6, 0.2, 0.7, 0.9, 0.4, -0.8]
bias = 7

#Menghitung output menggunakan fungsi dot produk, np.dot()
output = np.dot(weights, inputs) + bias

#Mencetak output
print(output)