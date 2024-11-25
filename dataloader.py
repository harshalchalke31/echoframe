import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class EchoDataset(Dataset):
    def __init__(self, video_dir, mask_dir, filelist_path, tracings_path, transform=None):
        """
        Initializes the dataset.

        Args:
        - video_dir (str): Path to the echocardiogram videos.
        - mask_dir (str): Path to the segmentation masks.
        - filelist_path (str): Path to the FileList.csv file.
        - tracings_path (str): Path to the VolumeTracings.csv file.
        - transform (callable, optional): Transform to apply to video frames.
        """
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.filelist = pd.read_csv(filelist_path)
        self.tracings = pd.read_csv(tracings_path)
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        # Fetch video and metadata
        video_name = self.filelist.iloc[idx]["VideoFileName"]
        video_path = os.path.join(self.video_dir, video_name)

        # Load video
        video = self._load_video(video_path)

        # Load segmentation masks
        mask_path = os.path.join(self.mask_dir, f"{os.path.splitext(video_name)[0]}_masks.npy")
        masks = np.load(mask_path)

        # Fetch ground truth frames from tracings
        gt_frames = self._get_ground_truth_frames(video_name)
        gt_masks = masks[gt_frames]

        # Prepare inputs: Current frame + previous frame's mask
        inputs, targets = [], []
        for i in range(1, len(video)):
            input_frame = video[i]
            prev_mask = masks[i - 1]
            inputs.append(np.concatenate([input_frame, prev_mask], axis=0))  # Concatenate along channel axis
            targets.append(masks[i])

        # Apply transforms if specified
        if self.transform:
            inputs = [self.transform(frame) for frame in inputs]
            targets = [self.transform(mask) for mask in targets]

        # Convert to tensors
        inputs = torch.tensor(np.stack(inputs), dtype=torch.float32)
        targets = torch.tensor(np.stack(targets), dtype=torch.long)  # Assuming masks are categorical

        return inputs, targets, gt_masks

    def _load_video(self, video_path):
        """
        Loads a video and returns it as a numpy array of shape [T, C, H, W].
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = np.transpose(frame, (2, 0, 1))  # Convert to [C, H, W]
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def _get_ground_truth_frames(self, video_name):
        """
        Retrieves ground truth frame indices for a given video.
        """
        gt_data = self.tracings[self.tracings["VideoFileName"] == video_name]
        return gt_data["FrameIndex"].values


def get_dataloader(video_dir, mask_dir, filelist_path, tracings_path, batch_size=8, shuffle=True, transform=None):
    """
    Creates a DataLoader for the dataset.

    Args:
    - video_dir (str): Path to the echocardiogram videos.
    - mask_dir (str): Path to the segmentation masks.
    - filelist_path (str): Path to the FileList.csv file.
    - tracings_path (str): Path to the VolumeTracings.csv file.
    - batch_size (int): Batch size for DataLoader.
    - shuffle (bool): Whether to shuffle the dataset.
    - transform (callable, optional): Transform to apply to video frames.

    Returns:
    - DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = EchoDataset(video_dir, mask_dir, filelist_path, tracings_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
