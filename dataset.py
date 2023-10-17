import copy
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import *
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from PIL import Image

def check_path_valid(path):
	return path if path.endswith('/') else path + '/'

def get_files(vec_folder):
    #得到当前vec_folder目录下的所有文件地址
    file_names = os.listdir(vec_folder)
    file_names.sort()
    vec_folder = check_path_valid(vec_folder)
    for i in range(len(file_names)):
        file_names[i] = vec_folder + file_names[i]
    return file_names

def load_data_xy(file_names):
    datas  = []
    labels = []
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            x, y = pickle.load(f, encoding='latin1')
        datas.append(x)
        labels.append(y)
    data_array = np.vstack(datas)
    label_array = np.hstack(labels)
    return data_array, label_array

class DDH_train_dataset(Dataset):
	def __init__(self, train_set_path):
		train_file_names = get_files(train_set_path)
		self.train_data, self.train_label = load_data_xy(train_file_names)
		#train_data = [n, 3027], train_label = [n, num_class]
		self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
		self.train_data = self.train_data.astype(np.uint8)
		#train_data = [n, 3, 32, 32]

		#将原有label进行深拷贝
		# self.gallery_set_y = copy.deepcopy(self.train_label)

		# encoder = OneHotEncoder(sparse_output=False)
		# self.train_label = encoder.fit_transform(self.train_label.reshape(-1, 1))

	def __len__(self):
		return len(self.train_data)

	def __getitem__(self, idx):
		data, label = self.train_data[idx], self.train_label[idx]

		data = data.transpose(1, 2, 0)
		# print(data.shape)
		transform = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		data = transform(data)
		label = torch.tensor(label, dtype=torch.long)
		return data, label
    
	def get_gallery_data(self):
		return self.train_data, self.train_label
	
class DDH_test_dataset(Dataset):
	def __init__(self, test_set_path):
		test_file_names = get_files(test_set_path)
		self.test_data, self.test_label = load_data_xy(test_file_names)
		self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
		self.test_data = self.test_data.astype(np.uint8)

		self.query_set_y = copy.deepcopy(self.test_label)

		encoder = OneHotEncoder(sparse_output=False)
		self.test_label = encoder.fit_transform(self.test_label.reshape(-1, 1))

	def __len__(self):
		return len(self.test_data)

	def __getitem__(self, idx):
		data, label = self.test_data[idx], self.test_label[idx]

		data = data.transpose(2, 1, 0)
		# print(data.shape)
		transform = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		data = transform(data)
		label = torch.from_numpy(label)
		return data, label

	def get_query_data(self):
		return self.test_data, self.query_set_y
	

if __name__ == "__main__":
	dataset = DDH_train_dataset(TRAIN_SET_PATH)
	dataloader = DataLoader(dataset, batch_size=6, shuffle=False)
	samples = iter(dataloader)
	data, label = next(samples)
	print(data.shape, label.shape)
	
	# print(data.shape)
	# print(label.shape)
	
	# #求label中的最小值
	# print(np.min(label))