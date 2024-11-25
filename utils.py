import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from data.SimLVSeg.simlvseg.utils import load_video, save_video
from data.SimLVSeg.simlvseg.seg_3d.pl_module import Seg3DModule


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




# def visualize_segmentation_masks(video_name, video_folder, filelist_path, tracings_path, num_frames=10):
#     """
#     Visualize the representative frames with segmentation masks from a video in the EchoNet-Pediatric dataset.
    
#     Parameters:
#         video_name (str): Name of the video file (e.g., 'your_sample_video.avi').
#         video_folder (str): Path to the folder containing video files.
#         tracings_path (str): Path to 'VolumeTracings.csv'.
#     """
#     # Load the volume tracings
#     tracings_df = pd.read_csv(tracings_path)

#     # Select tracings for the sample video and filter for unique frames
#     sample_tracings = tracings_df[tracings_df['FileName'] == video_name]
#     representative_frames = sample_tracings['Frame'].unique()

#     # Load the video
#     video_path = f"{video_folder}/{video_name}"
#     cap = cv2.VideoCapture(video_path)

#     # Prepare plot
#     num_frames = len(representative_frames)
#     fig, axes = plt.subplots(1, num_frames, figsize=(5 * num_frames, 5))
#     if num_frames == 1:
#         axes = [axes]  # Ensure axes is iterable even if there's only one frame

#     for i, frame_idx in enumerate(representative_frames):
#         # Set video to the specific frame position
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
#         ret, frame = cap.read()
        
#         if not ret:
#             print(f"Frame {frame_idx} could not be loaded, skipping.")
#             continue

#         # Get tracing coordinates for the current frame
#         frame_tracings = sample_tracings[sample_tracings['Frame'] == frame_idx]
#         x_coords = frame_tracings['X'].values
#         y_coords = frame_tracings['Y'].values

#         # Convert frame to RGB and plot
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         axes[i].imshow(frame_rgb)
        
#         # Overlay segmentation mask
#         axes[i].plot(x_coords, y_coords, 'r-', linewidth=2)
#         axes[i].set_title(f"Frame {frame_idx} (Representative)")
#         axes[i].axis('off')

#     cap.release()
#     plt.tight_layout()
#     plt.show()

# def visualize_segmentation_masks_with_background(video_name, video_folder, filelist_path, tracings_path):
#     """
#     Visualize the representative frames with segmentation masks from a video in the EchoNet-Pediatric dataset.
#     Additionally, draw the segmentation masks on blank plots with a background color for each frame separately.
    
#     Parameters:
#         video_name (str): Name of the video file (e.g., 'your_sample_video.avi').
#         video_folder (str): Path to the folder containing video files.
#         tracings_path (str): Path to 'VolumeTracings.csv'.
#     """
#     # Load the volume tracings
#     tracings_df = pd.read_csv(tracings_path)

#     # Select tracings for the sample video and filter for unique frames
#     sample_tracings = tracings_df[tracings_df['FileName'] == video_name]
#     representative_frames = sample_tracings['Frame'].unique()

#     # Load the video
#     video_path = f"{video_folder}/{video_name}"
#     cap = cv2.VideoCapture(video_path)

#     for frame_idx in representative_frames:
#         # Set video to the specific frame position
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
#         ret, frame = cap.read()
        
#         if not ret:
#             print(f"Frame {frame_idx} could not be loaded, skipping.")
#             continue

#         # Get tracing coordinates for the current frame
#         frame_tracings = sample_tracings[sample_tracings['Frame'] == frame_idx]
#         x_coords = frame_tracings['X'].values
#         y_coords = frame_tracings['Y'].values

#         # Convert frame to RGB and plot the original frame with segmentation overlay
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Plot the original frame with segmentation overlay
#         fig1, ax1 = plt.subplots(figsize=(5, 5))
#         ax1.imshow(frame_rgb)
#         ax1.plot(x_coords, y_coords, 'r-', linewidth=2)
#         ax1.set_title(f"Frame {frame_idx}")
#         ax1.axis('off')
#         plt.show()

#         # Draw the segmentation mask on a blank plot with background color
#         fig2, ax2 = plt.subplots(figsize=(5, 5))

#         # Draw a filled rectangle to set the background color manually
#         ax2.add_patch(plt.Rectangle((0, 0), frame_rgb.shape[1], frame_rgb.shape[0], color="#2C3E50"))

#         # Fill the segmentation area in red
#         ax2.fill(x_coords, y_coords, '#F7DC6F')

#         # Set the limits to match the frame dimensions and invert y-axis
#         ax2.set_xlim([0, frame_rgb.shape[1]])
#         ax2.set_ylim([frame_rgb.shape[0], 0])

#         # Turn off the axes for a cleaner look
#         ax2.axis('off')
#         ax2.set_title(f"Segmentation Mask (Frame {frame_idx})")

#         plt.show()


#     cap.release()





# def visualize_segmentation_for_adults(video_name, video_folder, filelist_path, tracings_path):
#     """
#     Visualize the segmentation masks for both end-systolic and end-diastolic frames based on available frames in VolumeTracings.csv.

#     Parameters:
#         video_name (str): Name of the video file (e.g., 'sample_video.avi').
#         video_folder (str): Path to the folder containing video files.
#         filelist_path (str): Path to the FileList.csv.
#         tracings_path (str): Path to the VolumeTracings.csv.
#     """
#     # Load the file list and volume tracings
#     file_list_df = pd.read_csv(filelist_path)
#     volume_tracings_df = pd.read_csv(tracings_path)

#     # Remove .avi extension from the video name for consistent comparison
#     video_name_no_ext = video_name.replace('.avi', '')

#     # Filter the file list for the given video
#     video_info = file_list_df[file_list_df['FileName'] == video_name_no_ext]
#     if video_info.empty:
#         print(f"Video {video_name} not found in FileList.csv.")
#         return

#     # Access frame height and width properly using .iloc
#     frame_height = int(video_info['FrameHeight'].iloc[0])
#     frame_width = int(video_info['FrameWidth'].iloc[0])

#     # Filter volume tracings for the given video
#     video_tracings = volume_tracings_df[volume_tracings_df['FileName'] == video_name]

#     # Check if there are tracings available
#     if video_tracings.empty:
#         print("Segmentation coordinates for the specified video not found.")
#         return

#     # Identify the unique frames available in tracings (e.g., end-systolic and end-diastolic)
#     unique_frames = video_tracings['Frame'].unique()
#     if len(unique_frames) < 2:
#         print("Insufficient frames for end-systolic and end-diastolic segmentation.")
#         return

#     # Assume the smallest frame number as end-diastolic and the largest as end-systolic
#     end_diastolic_frame_num = unique_frames.min()
#     end_systolic_frame_num = unique_frames.max()

#     # Extract the tracings for these frames
#     end_diastolic_frame = video_tracings[video_tracings['Frame'] == end_diastolic_frame_num]
#     end_systolic_frame = video_tracings[video_tracings['Frame'] == end_systolic_frame_num]

#     # Create masks for the frames
#     def create_mask(tracings, height, width):
#         mask = np.zeros((height, width), dtype=np.uint8)
#         for _, row in tracings.iterrows():
#             x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
#             mask[y1:y2, x1:x2] = 1  # Fill the region in the mask
#         return mask

#     systolic_mask = create_mask(end_systolic_frame, frame_height, frame_width)
#     diastolic_mask = create_mask(end_diastolic_frame, frame_height, frame_width)

#     # Plot the masks
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(systolic_mask, cmap='viridis')
#     axs[0].set_title(f'End-Systolic Mask (Frame {end_systolic_frame_num})')
#     axs[0].axis('off')

#     axs[1].imshow(diastolic_mask, cmap='viridis')
#     axs[1].set_title(f'End-Diastolic Mask (Frame {end_diastolic_frame_num})')
#     axs[1].axis('off')

#     plt.show()

# # Example call to the function (assuming video is in the specified folder and files are available)
# # visualize_segmentation_for_systolic_diastolic("0X100009310A3BD7FC.avi", "/path/to/video_folder", file_list_path, volume_tracings_path)

# # Usage
# if __name__ == "__main__":
#     visualize_segmentation_masks('your_sample_video.avi', 'Videos')
