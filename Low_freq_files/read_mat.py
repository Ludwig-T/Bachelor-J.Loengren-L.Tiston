import scipy.io
import os

folder_path = 'C:/Data/label_low_freq'

def read_mat(file_path):
    data = scipy.io.loadmat(file_path)
    print(len(data['l1'][0][0][0]))

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    data = scipy.io.loadmat(file_path)
    print(data['l1'][0][0][4][0][0][0][0][4])
    break
    