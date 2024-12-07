import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
import numpy as np

class EchoDynamicDataset(Dataset):
    def __init__(self, 
                 root="./data/echodynamic", 
                 split='train', 
                 resize=(112,112),
                 mean=[0.0,0.0,0.0],
                 std=[1.0,1.0,1.0]):
        """
        Args:
            root (str): Path to the echodynamic data directory.
            split (str): 'train', 'val', or 'test'.
            resize (tuple): Desired (H,W) of frames and masks.
            mean (list): Normalization mean for RGB channels.
            std (list): Normalization std for RGB channels.
        """
        self.root = root
        self.split = split
        self.resize = resize
        self.mean = mean
        self.std = std

        filelist_path = os.path.join(self.root, "FileList.csv")
        df = pd.read_csv(filelist_path)
        df = df[df['Split'].str.lower() == self.split]
        
        # Extract the filenames of videos
        self.fnames = df['FileName'].apply(lambda x: os.path.splitext(x)[0] + ".avi").values

        # Build a list of (video, frame_idx) samples for ALL frames in each video
        self.samples = []
        for fname in self.fnames:
            video_path = os.path.join(self.root, "Videos", fname)
            if not os.path.exists(video_path):
                continue
            # Count frames in this video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Add a sample for each frame
            for fidx in range(frame_count):
                self.samples.append((fname, fidx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, fidx = self.samples[idx]

        # Load current frame (3,H,W)
        frame = self._load_frame(fname, fidx)

        # Load previous mask or zero (1,H,W)
        if fidx == 0:
            prev_mask = np.zeros((1, *self.resize), dtype=np.float32)
        else:
            prev_mask = self._load_mask(fname, fidx-1)

        # Load current mask (1,H,W)
        cur_mask = self._load_mask(fname, fidx)

        # Combine current frame (3 channels) and prev_mask (1 channel) -> (4,H,W)
        inp = np.concatenate([frame, prev_mask], axis=0) # shape: (4,112,112)
        inp = torch.from_numpy(inp).float()

        cur_mask = torch.from_numpy(cur_mask).float()    # shape: (1,112,112)

        return inp, cur_mask

    def _load_frame(self, fname, fidx):
        """
        Load and return frame fidx of video fname as a (3,H,W) numpy array normalized and resized.
        """
        video_path = os.path.join(self.root, "Videos", fname)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # If we fail to read, return a zero image
            frame = np.zeros((self.resize[0], self.resize[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)

        frame = frame.astype(np.float32)/255.0
        # Normalize if desired
        for c in range(3):
            frame[:,:,c] = (frame[:,:,c] - self.mean[c]) / self.std[c]

        # transpose to (C,H,W)
        frame = np.transpose(frame, (2,0,1)) # (3,H,W)
        return frame

    def _load_mask(self, fname, fidx):
        """
        Load mask tensor (single-channel) from augmented_masks.
        Mask files are named {video_name}_frame{fidx}.pt.
        """
        video_name = os.path.splitext(fname)[0]  # remove .avi
        mask_path = os.path.join(self.root, "augmented_masks", f"{video_name}_frame{fidx}.pt")

        if not os.path.exists(mask_path):
            # If mask doesn't exist, return zero
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            m = torch.load(mask_path) # (H,W)
            mask = m.numpy().astype(np.float32)

            # Resize mask if needed
            if mask.shape != self.resize:
                mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension: (1,H,W)
        mask = np.expand_dims(mask, axis=0)
        return mask

if __name__ == "__main__":
    # Example usage
    dataset = EchoDynamicDataset(root="./data/echodynamic", split='train')
    inp, mask = dataset[0]
    print("Input shape:", inp.shape)   # (4,112,112)
    print("Mask shape:", mask.shape)   # (1,112,112)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Total samples:", len(dataset))
    print("Total batches (batch_size=1):", len(loader))

    # Check one batch from the DataLoader
    for batch_inp, batch_mask in loader:
        print("Batch input shape:", batch_inp.shape)  # (B,4,H,W)
        print("Batch mask shape:", batch_mask.shape)  # (B,1,H,W)
        break
