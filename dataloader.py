import torch
import os
import pandas as pd
import numpy as np
import collections
import cv2
import skimage.draw
import matplotlib.pyplot as plt


class EchoMasks(torch.utils.data.Dataset):
	def __init__(self, root = "./data/echodynamic/",split='train', batch_size = 1, noise = None, padding = None):
		self.folder = root
		self.split = split
		self.batch_size = batch_size
		self.noise = noise
		self.pad = padding

		self.fnames =[]
		self.outcomes =[]
		self.ejection=[]
		self.fps = []

		with open(self.folder + "FileList.csv") as f:
			header   = f.readline().strip().split(",")
			filenameIndex = header.index("FileName")
			splitIndex = header.index("Split")
			efIndex = header.index("EF")
			fpsIndex = header.index("FPS")
			for line in f:
				lineSplit = line.strip().split(',')
				# Get name of the video file
				fileName = os.path.splitext(lineSplit[filenameIndex])[0]+".avi"
				
				#Get the subset that the video belongs to 
				fileSet = lineSplit[splitIndex].lower()
				
				#Get ef for the video
				fileEf = lineSplit[efIndex]

				#Get fps for the video
				fileFps = lineSplit[fpsIndex]

				#Ensure that the video exists 
				if os.path.exists(self.folder + "/Videos/" + fileName):
					if fileSet == split:
						self.fnames.append(fileName)
						self.outcomes.append(lineSplit)
						self.ejection.append(fileEf)
						self.fps.append(fileFps)

		self.frames = collections.defaultdict(list)
		_defaultdict_of_lists_ = collections.defaultdict(list)
		self.trace = collections.defaultdict(lambda: collections.defaultdict(list))

		#Read the voilume tracings CSV file to find videos with ED/ES frames 
		with open(self.folder + "VolumeTracings.csv") as f:
			header = f.readline().strip().split(",")
			assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

			for line in f:
				filename, x1, y1, x2, y2, frame = line.strip().split(",")
				x1 = float(x1)
				x2 = float(x2)
				y1 = float(y1)
				y2 = float(y2)
				frame = int(frame)

				# New frame index for the given filename
				if frame not in self.trace[filename]:
					self.frames[filename].append(frame)
				self.trace[filename][frame].append((x1,y1,x2,y2))

		#Transform into numpy array 
		for filename in self.frames:
			for frame in self.frames[filename]:
				self.trace[filename][frame] = np.array(self.trace[filename][frame])

		# Reject files without tracings 
		keep = [len(self.frames[f]) >= 2 for f in self.fnames]
		self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
		self.outcomes = [f for (f, k) in zip(self.outcomes, keep) if k]

		self.indexes = np.arange(np.shape(self.fnames)[0])


	def __len__(self):
		# Denotes the number of batches per epoch
		return(int (np.floor(np.shape(self.fnames)[0])/self.batch_size))

	def __getitem__(self, idx):
		# Generate one batch of data
		# Generate indexes of the batch

		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		X, y = self.__data_generation(indexes)


		return X, y


	def __data_generation(self, list_IDs_temp):

		X = []
		y = []

		index = 0
		for i in list_IDs_temp:
			path = os.path.join(self.folder, "Videos", self.fnames[i])
			# Load video into np.array
			if not os.path.exists(path):
				print("File does not exist")

			frames = self.frames[self.fnames[i]]
			frames.sort() #  Ensure that the frames are in ascending order
			traces = self.trace[self.fnames[i]]

			vid_cap = cv2.VideoCapture(path)
			frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

			# Read the entire video and save the traced frames 
			inputs = np.zeros((len(frames),frame_width, frame_height, 3), np.uint8)
			targets = np.zeros((len(frames),frame_width, frame_height), np.uint8)
			index = 0
			
			# Load the frames  
			for count in range(frame_count): 
				success, frame = vid_cap.read()
				if not success:
					print("Failed to load frame #", count, ' of ', self.fnames[i])
				
				if (count) in frames: #Assume that frame number is 1 indexed
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					inputs[index] = frame
					index = index + 1
			
			# blackout pixels for simulated noise
			if self.noise:
				num_pepper = np.ceil(self.noise * 2 * frame_height * frame_width)
				coords = [np.random.randint(0, i-1, int(num_pepper)) for i in inputs.shape[0:3]]
				inputs[tuple(coords)] = (0,0,0)
			
			# Scale pixels between 0 and 1
			inputs = inputs /255.0
			X = np.append(X, inputs)
			X = X.reshape((-1, 112,112,3))
	
			#Load the targets
			index = 0
			for f in frames:
				t = traces[f]
				x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
				x = np.concatenate((x1[1:], np.flip(x2[1:])))
				y_ = np.concatenate((y1[1:], np.flip(y2[1:])))
				r, c = skimage.draw.polygon(np.rint(y_).astype(int), np.rint(x).astype(int), (frame_width,frame_height))
				mask = np.zeros((frame_width, frame_height), np.float32)
				mask[r, c] = 1
				targets[index] = mask
				index = index + 1	

			y = np.append(y, targets)
			y = y.reshape(-1, 112, 112)

		y = np.expand_dims(y, -1)
		
		if self.pad is not None:
			X = np.pad(X, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='constant', constant_values=0)
			y = np.pad(y, ((0,0),(self.pad,self.pad),(self.pad,self.pad), (0,0)), mode='constant', constant_values=0)

		return X, y

	def display_example(self, idx):
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		X, y = self.__data_generation(indexes)

		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
		axs[0].imshow(X[0])
		axs[0].set_title("Frame")
		axs[1].imshow(y[0])
		axs[1].set_title("Mask")
		plt.show()
