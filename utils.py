import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
# from data.SimLVSeg.simlvseg.utils import load_video, save_video
# from data.SimLVSeg.simlvseg.seg_3d.pl_module import Seg3DModule
import torch.nn as nn

def get_config(name:str):
    HE = nn.Hardswish()
    RE = nn.ReLU()
    large = [
        [3, 16, 16, 16, False, RE, 1],
        [3, 64, 16, 24, False, RE, 2],
        [3, 72, 24, 24, False, RE, 1],
        [5, 72, 24, 40, True, RE, 2],
        [5, 120, 40, 40, True, RE, 1],
        [5, 120, 40, 40, True, RE, 1],
        [3, 240, 40, 80, False, HE, 2],
        [3, 200, 80, 80, False, HE, 1],
        [3, 184, 80, 80, False, HE, 1],
        [3, 184, 80, 80, False, HE, 1],
        [3, 480, 80, 112, True, HE, 1],
        [3, 672, 112, 112, True, HE, 1],
        [5, 672, 112, 160, True, HE, 2],
        [5, 960, 160, 160, True, HE, 1],
        [5, 960, 160, 160, True, HE, 1],
    ]
    small = [
        [3, 16, 16, 16, True, RE, 2],
        [3, 72, 16, 24, False, RE, 2],
        [3, 88, 24, 24, False, RE, 1],
        [5, 96, 24, 40, True, HE, 2],
        [5, 240, 40, 40, True, HE, 1],
        [5, 240, 40, 40, True, HE, 1],
        [5, 120, 40, 48, True, HE, 1],
        [5, 144, 48, 48, True, HE, 1],
        [5, 288, 48, 96, True, HE, 2],
        [5, 576, 96, 96, True, HE, 1],
        [5, 576, 96, 96, True, HE, 1],
    ]
    if name == "large":
        return large
    elif name == "small":
        return small
    else:
        raise ValueError("config_name must be 'large' or 'small'.")


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = dice_loss

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch_inp, batch_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # batch_inp: (1, num_frames, 4, H, W)
            # batch_mask: (1, num_frames, 1, H, W)
            batch_inp = batch_inp.squeeze(0).to(device)   # (num_frames,4,H,W)
            batch_mask = batch_mask.squeeze(0).to(device) # (num_frames,1,H,W)

            # Forward pass with 4 channels
            # Model expects (B,C,H,W), and we have (num_frames,4,H,W) which is compatible directly as B=num_frames
            pred_mask = model(batch_inp)  # (num_frames,1,H,W)

            loss = criterion(pred_mask, batch_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_inp, batch_mask in val_loader:
                batch_inp = batch_inp.squeeze(0).to(device)
                batch_mask = batch_mask.squeeze(0).to(device)
                pred_mask = model(batch_inp)
                loss = criterion(pred_mask, batch_mask)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

    return model

# class InferenceDataset(torch.utils.data.Dataset):
#     def __init__(self, vid_path, length, period, preprocessing):
#         self.vid_path = vid_path
#         self.video = load_video(vid_path).transpose((3, 0, 1, 2))
#         self.fps = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)

#         self.length = length
#         self.period = period
#         self.preprocessing = preprocessing

#         c, f, h, w = self.video.shape
#         if f < length * self.period:
#             self.video = np.concatenate((self.video, np.zeros((c, length * self.period - f, h, w), self.video.dtype)), axis=1)

#         self.list_selected_frame_indexes = []
#         pointer = 0
#         inner_loop = True
#         while True:
#             for i in range(self.period):
#                 if not inner_loop:
#                     break

#                 start = pointer + i
#                 selected_frame_indexes = start + self.period * np.arange(length)

#                 if selected_frame_indexes[-1] >= f:
#                     inner_loop = False
#                     break

#                 self.list_selected_frame_indexes.append(selected_frame_indexes)

#             if self.list_selected_frame_indexes[-1][-1] == f - 1:
#                 break

#             pointer = min(pointer + length * self.period, f - (length - 1) * self.period - self.period)

#     def __len__(self):
#         return len(self.list_selected_frame_indexes)

#     def __getitem__(self, idx):
#         selected_frame_indexes = self.list_selected_frame_indexes[idx]
#         video = self.video[:, selected_frame_indexes, :, :].copy()
#         video = video.transpose((1, 2, 3, 0))
#         video, _ = self.preprocessing(video)
#         video = video.transpose((1, 2, 3, 0))
#         return video, selected_frame_indexes




