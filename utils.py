import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data.SimLVSeg.simlvseg.utils import load_video, save_video
from data.SimLVSeg.simlvseg.seg_3d.pl_module import Seg3DModule
import torch.nn as nn


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, vid_path, length, period, preprocessing):
        self.vid_path = vid_path
        self.video = load_video(vid_path).transpose((3, 0, 1, 2))
        self.fps = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)

        self.length = length
        self.period = period
        self.preprocessing = preprocessing

        c, f, h, w = self.video.shape
        if f < length * self.period:
            self.video = np.concatenate((self.video, np.zeros((c, length * self.period - f, h, w), self.video.dtype)), axis=1)

        self.list_selected_frame_indexes = []
        pointer = 0
        inner_loop = True
        while True:
            for i in range(self.period):
                if not inner_loop:
                    break

                start = pointer + i
                selected_frame_indexes = start + self.period * np.arange(length)

                if selected_frame_indexes[-1] >= f:
                    inner_loop = False
                    break

                self.list_selected_frame_indexes.append(selected_frame_indexes)

            if self.list_selected_frame_indexes[-1][-1] == f - 1:
                break

            pointer = min(pointer + length * self.period, f - (length - 1) * self.period - self.period)

    def __len__(self):
        return len(self.list_selected_frame_indexes)

    def __getitem__(self, idx):
        selected_frame_indexes = self.list_selected_frame_indexes[idx]
        video = self.video[:, selected_frame_indexes, :, :].copy()
        video = video.transpose((1, 2, 3, 0))
        video, _ = self.preprocessing(video)
        video = video.transpose((1, 2, 3, 0))
        return video, selected_frame_indexes




