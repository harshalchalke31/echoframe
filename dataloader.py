import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import collections
import cv2

class EchoMasks(Dataset):
    def __init__(self, root="./data/", split='train', transform=None):
        self.folder = root
        self.split = split
        self.transform = transform

        self.fnames = []
        self.ejection = []
        self.fps = []

        # read the file list CSV
        file_list = pd.read_csv(os.path.join(self.folder, "FileList.csv"))
        for index, row in file_list.iterrows():
            file_name = os.path.splitext(row["FileName"])[0] + ".avi"
            file_set = row["Split"].lower()
            file_ef = row["EF"]
            file_fps = row["FPS"]

            # ensure the video file exists and matches the split
            if os.path.exists(os.path.join(self.folder, "Videos", file_name)):
                if file_set == split:
                    self.fnames.append(file_name)
                    self.ejection.append(float(file_ef))
                    self.fps.append(file_fps)

        # load tracing information for ED/ES frames
        self.trace = collections.defaultdict(lambda: collections.defaultdict(list))
        tracings = pd.read_csv(os.path.join(self.folder, "VolumeTracings.csv"))
        for index, row in tracings.iterrows():
            file_name, x1, y1, x2, y2, frame = row["FileName"], row["X1"], row["Y1"], row["X2"], row["Y2"], row["Frame"]
            self.trace[file_name][frame].append([float(x1), float(y1), float(x2), float(y2)])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # get the video path and load video frames
        video_path = os.path.join(self.folder, "Videos", self.fnames[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # convert to tensor
        frames_tensor = torch.stack([torch.tensor(f, dtype=torch.float32) for f in frames])

        # get the ejection fraction for the video
        ejection_fraction = torch.tensor(self.ejection[idx], dtype=torch.float32)

        # get frame tracing data
        trace = self.trace[self.fnames[idx]]

        return {"video": frames_tensor, "ejection_fraction": ejection_fraction, "trace": trace}

    @staticmethod
    def custom_collate_fn(batch):
        # each item in batch is a dict of {'video': Tensor, 'ejection_fraction': Tensor, 'trace': dict}
        batch_videos = [item['video'] for item in batch]
        batch_ef = torch.stack([item['ejection_fraction'] for item in batch])
        batch_trace = [item['trace'] for item in batch]

        return {"videos": batch_videos, "ejection_fractions": batch_ef, "traces": batch_trace}

    def get_dataloader(self, batch_size=2, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.custom_collate_fn)

def load_data():
    dataset = EchoMasks(root="EchoNet-Dynamic/", split='train')
    dataloader = dataset.get_dataloader(batch_size=2, shuffle=True)

    for batch in dataloader:
        print(f"Batch videos: {len(batch['videos'])}")
        print(f"Ejection fractions: {batch['ejection_fractions']}")
        print(f"Trace data: {batch['traces']}")
        break

# Call the function to load the data
if __name__ == "__main__":
    load_data()
