import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def build_polygons_dict(volume_csv_path):
    """
    Gathers partial tracing rows from VolumeTracings.csv into a dict:
      (base_name, frame_number) -> [ [(x1,y1), (x2,y2), ... ] ]
    """
    tracings_df = pd.read_csv(volume_csv_path)
    
    grouped = tracings_df.groupby(["FileName", "Frame"])
    polygons_dict = {}
    for (filename, frame), group_rows in grouped:
        coords = []
        for _, row in group_rows.iterrows():
            x1, y1 = float(row["X1"]), float(row["Y1"])
            x2, y2 = float(row["X2"]), float(row["Y2"])
            coords.append((x1, y1))
            coords.append((x2, y2))

        base_name = os.path.splitext(filename)[0]
        key = (base_name, int(frame))
        polygons_dict[key] = [coords]
    return polygons_dict

def sort_polygon_coords(coords):
    """
    Sort (x,y) points by angle around the centroid, forming a continuous path.
    """
    centroid = np.mean(coords, axis=0)
    angles = [np.arctan2(y - centroid[1], x - centroid[0]) for (x, y) in coords]
    sorted_idx = np.argsort(angles)
    return [coords[i] for i in sorted_idx]

def create_mask(polygons_dict, video_name, frame_idx, hw_shape):
    """
    Builds a binary 2D mask of shape hw_shape for (video_name, frame_idx).
    Applies a morphological 'close' to fill small holes/gaps.
    Returns a mask in range {0,1}.
    """
    base_name = os.path.splitext(video_name)[0]
    key = (base_name, frame_idx)

    mask = np.zeros(hw_shape, dtype=np.uint8)
    if key not in polygons_dict:
        return mask  # No polygons => empty mask

    for polygon_coords in polygons_dict[key]:
        coords_sorted = sort_polygon_coords(polygon_coords)
        pts = np.array(coords_sorted, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    # Morphological closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = (mask > 127).astype(np.uint8)
    return mask

def read_frame_resized(video_path, frame_idx, resize=(112, 112)):
    """
    Reads the 0-based frame_idx from the video, converts BGR->RGB,
    and resizes to (width,height) if desired. Returns (H,W,3) in RGB.
    If the read fails, returns None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if resize is not None:
        w, h = resize  # (width, height)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


class EchoMaskDataset(Dataset):
    def __init__(self, 
                 df,                # DataFrame with at least "FileName" column
                 polygons_dict,     # from build_polygons_dict(...)
                 videos_path,       # path to .avi files
                 transform=None,    # optional transform pipeline for (image, mask)
                 resize=(112, 112),
                 all_traced_frames=True,
                 offset_frames_by_one=False):
        """
        PyTorch dataset for image+mask segmentation on EchoNet data.

        Args:
          df: Pandas DataFrame with 'FileName' column and possibly others (train/val/test)
          polygons_dict: dict from build_polygons_dict(...) => (base_name, frame_idx)->[...]
          videos_path: folder where the .avi files reside
          transform: optional callable for augmentations; it should accept
                     and return (image, mask) as Tensors
          resize:  (width, height) for resizing frames & masks
          all_traced_frames: if True, use every traced frame from polygons_dict
                             if False, use only min (ES) and max (ED) frames
          offset_frames_by_one: if your CSV uses 1-based frames but OpenCV is 0-based,
                                set this True to do (frame_idx - 1).
        """
        self.df = df.reset_index(drop=True)
        self.polygons_dict = polygons_dict
        self.videos_path = videos_path
        self.transform = transform
        self.resize = resize
        self.all_traced_frames = all_traced_frames
        self.offset_frames_by_one = offset_frames_by_one

        # 1) Build list of (FileName, frame_idx) samples
        self.samples = []
        
        # Convert polygons_dict keys to a list of (BaseName, Frame)
        # E.g. polygons_dict keys: (base_name, frame_number)
        all_keys = list(polygons_dict.keys())
        # We'll build a quick dictionary: base_name -> list of frame_numbers
        base_frames_map = {}
        for (bn, fr) in all_keys:
            if bn not in base_frames_map:
                base_frames_map[bn] = []
            base_frames_map[bn].append(fr)

        # 2) For each file in df, gather frames from base_frames_map
        for _, row in self.df.iterrows():
            file_name = row["FileName"]  # e.g. 0X100009310A3BD7FC.avi
            base_name = os.path.splitext(file_name)[0]

            if base_name not in base_frames_map:
                continue  # no annotated frames for this file

            frames_this_file = base_frames_map[base_name]

            if len(frames_this_file) == 0:
                continue

            # If we want all traced frames, add them all
            if self.all_traced_frames:
                for fr in frames_this_file:
                    self.samples.append((file_name, fr))
            else:
                # Just the min and max frames for that file
                es_frame = min(frames_this_file)
                ed_frame = max(frames_this_file)
                self.samples.append((file_name, es_frame))
                if ed_frame != es_frame:
                    self.samples.append((file_name, ed_frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, frame_idx = self.samples[idx]
        video_path = os.path.join(self.videos_path, file_name)

        # If your VolumeTracings.csv is 1-based and OpenCV expects 0-based,
        # offset frames by 1
        if self.offset_frames_by_one:
            frame_idx -= 1
            if frame_idx < 0:
                frame_idx = 0

        # 1) Load the image => shape (H,W,3)
        image_np = read_frame_resized(video_path, frame_idx, resize=self.resize)
        if image_np is None:
            # If something fails, fallback to black image
            w, h = self.resize
            image_np = np.zeros((h, w, 3), dtype=np.uint8)

        # 2) Create the mask => shape (H,W)
        mask_np = create_mask(
            polygons_dict=self.polygons_dict,
            video_name=file_name,
            frame_idx=frame_idx,
            hw_shape=image_np.shape[:2]  # (H,W)
        )

        # mask_np is 0 or 1 => shape (H,W)
        # Expand to (H,W,1)
        mask_np = np.expand_dims(mask_np, axis=-1)

        # 3) Convert to PyTorch Tensors, reorder to CHW
        # image: (3,H,W), mask: (1,H,W)
        image_tensor = torch.from_numpy(image_np).permute(2,0,1).float() / 255.0
        mask_tensor  = torch.from_numpy(mask_np).permute(2,0,1).float()

        # 4) Optional transforms
        if self.transform is not None:
            # We'll assume `transform` is a callable that takes in
            #  (image_tensor, mask_tensor) and returns (image_tensor, mask_tensor).
            # This is a common approach with custom joint augmentations.
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
