import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EchoVideoDataset(Dataset):
    def __init__(self, 
                 root="./data/echodynamic", 
                 split='train', 
                 resize=(112,112),
                 mean=[0.0,0.0,0.0],
                 std=[1.0,1.0,1.0]):
        """
        Args:
            root (str): Path to echodynamic data dir.
            split (str): 'train', 'val', or 'test'.
            resize (tuple): (H, W) for resizing frames/masks.
            mean (list): Normalization mean for RGB.
            std (list): Normalization std for RGB.
        """
        self.root = root
        self.split = split
        self.resize = resize
        self.mean = mean
        self.std = std

        filelist_path = os.path.join(self.root, "FileList.csv")
        df = pd.read_csv(filelist_path)
        df = df[df['Split'].str.lower() == self.split]

        # Extract video names (without .avi)
        self.video_names = df['FileName'].apply(lambda x: os.path.splitext(x)[0]).values

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.root, "Videos", video_name + ".avi")

        # IMPORTANT: Adjust this depending on your mask file names
        # If your working code loads without .pt extension, remove it here.
        # mask_path = os.path.join(self.root, "augmented_masks", video_name + ".pt")
        mask_path = os.path.join(self.root, "augmented_masks", video_name)  # Without ".pt"

        # Load video frames
        frames = self._load_all_frames(video_path)

        # Load masks
        masks = self._load_all_masks(mask_path, len(frames))

        # Build inputs and targets
        num_frames = len(frames)
        inputs = []
        targets = []

        for i in range(num_frames):
            current_frame = frames[i]  # (3,H,W)
            if i == 0:
                prev_mask = np.zeros((1,*self.resize), dtype=np.float32)
            else:
                prev_mask = masks[i-1]  # (1,H,W)

            cur_mask = masks[i]  # (1,H,W)
            inp = np.concatenate([current_frame, prev_mask], axis=0)  # (4,H,W)
            inputs.append(inp)
            targets.append(cur_mask)

        inputs = np.stack(inputs, axis=0)   # (num_frames,4,H,W)
        targets = np.stack(targets, axis=0) # (num_frames,1,H,W)

        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        return inputs, targets

    def _load_all_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32)/255.0
            # Normalize
            for c in range(3):
                frame[:,:,c] = (frame[:,:,c] - self.mean[c]) / self.std[c]
            frame = np.transpose(frame, (2,0,1)) # (3,H,W)
            frames.append(frame)
        cap.release()
        return frames

    def _load_all_masks(self, mask_path, expected_frames):
        if not os.path.exists(mask_path):
            # If no mask file, create zero masks
            masks = np.zeros((expected_frames, *self.resize), dtype=np.float32)
            masks = np.expand_dims(masks, axis=1)  # (num_frames,1,H,W)
            return masks

        # Load the mask tensor (num_frames,H,W)
        m = torch.load(mask_path)
        masks = m.numpy().astype(np.float32)

        # Check frame count
        if masks.shape[0] != expected_frames:
            min_frames = min(masks.shape[0], expected_frames)
            new_masks = np.zeros((expected_frames, masks.shape[1], masks.shape[2]), dtype=np.float32)
            new_masks[:min_frames] = masks[:min_frames]
            masks = new_masks

        # Resize masks and add channel dimension
        resized_masks = []
        for i in range(masks.shape[0]):
            mask_frame = masks[i]  # (H,W)
            if mask_frame.shape != self.resize:
                mask_frame = cv2.resize(mask_frame, self.resize, interpolation=cv2.INTER_NEAREST)
            # Ensure mask is binary (0 or 1). If needed, threshold:
            # mask_frame = (mask_frame > 0.5).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=0)  # (1,H,W)
            resized_masks.append(mask_frame)
        masks = np.stack(resized_masks, axis=0) # (num_frames,1,H,W)

        return masks

if __name__ == "__main__":
    # Testing code
    echo_train = EchoVideoDataset(root="./data/echodynamic", split='train')
    train_loader = DataLoader(echo_train, batch_size=1, shuffle=True)

    print("Total videos (batches) in train split:", len(echo_train))
    for batch_inp, batch_mask in train_loader:
        batch_inp = batch_inp.squeeze(0)   # (num_frames,4,H,W)
        batch_mask = batch_mask.squeeze(0) # (num_frames,1,H,W)

        print("Batch input shape:", batch_inp.shape)
        print("Batch mask shape:", batch_mask.shape)
        
        # Check min/max of a mask to ensure it's not all zeros
        print("Mask min:", batch_mask.min().item(), "Mask max:", batch_mask.max().item())

        # Visualize first 8 frames
        num_frames = batch_inp.shape[0]
        max_frames_to_show = min(num_frames, 8)

        fig, axs = plt.subplots(nrows=max_frames_to_show, ncols=2, figsize=(6, 3*max_frames_to_show))
        for i in range(max_frames_to_show):
            current_frame = batch_inp[i,:3].numpy().transpose(1,2,0)
            current_mask = batch_mask[i,0].numpy()

            axs[i,0].imshow(current_frame)
            axs[i,0].set_title(f"Frame {i} - RGB")
            axs[i,0].axis('off')

            axs[i,1].imshow(current_mask, cmap='gray')
            axs[i,1].set_title(f"Frame {i} - Mask")
            axs[i,1].axis('off')

        plt.tight_layout()
        plt.show()
        break  # Display only one batch
