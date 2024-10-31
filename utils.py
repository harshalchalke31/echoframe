import cv2
import pandas as pd
import matplotlib.pyplot as plt


def visualize_segmentation_masks(video_name, video_folder, filelist_path, tracings_path, num_frames=10):
    """
    Visualize the representative frames with segmentation masks from a video in the EchoNet-Pediatric dataset.
    
    Parameters:
        video_name (str): Name of the video file (e.g., 'your_sample_video.avi').
        video_folder (str): Path to the folder containing video files.
        tracings_path (str): Path to 'VolumeTracings.csv'.
    """
    # Load the volume tracings
    tracings_df = pd.read_csv(tracings_path)

    # Select tracings for the sample video and filter for unique frames
    sample_tracings = tracings_df[tracings_df['FileName'] == video_name]
    representative_frames = sample_tracings['Frame'].unique()

    # Load the video
    video_path = f"{video_folder}/{video_name}"
    cap = cv2.VideoCapture(video_path)

    # Prepare plot
    num_frames = len(representative_frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(5 * num_frames, 5))
    if num_frames == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one frame

    for i, frame_idx in enumerate(representative_frames):
        # Set video to the specific frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        
        if not ret:
            print(f"Frame {frame_idx} could not be loaded, skipping.")
            continue

        # Get tracing coordinates for the current frame
        frame_tracings = sample_tracings[sample_tracings['Frame'] == frame_idx]
        x_coords = frame_tracings['X'].values
        y_coords = frame_tracings['Y'].values

        # Convert frame to RGB and plot
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(frame_rgb)
        
        # Overlay segmentation mask
        axes[i].plot(x_coords, y_coords, 'r-', linewidth=2)
        axes[i].set_title(f"Frame {frame_idx} (Representative)")
        axes[i].axis('off')

    cap.release()
    plt.tight_layout()
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_segmentation_for_adults(video_name, video_folder, filelist_path, tracings_path):
    """
    Visualize the segmentation masks for both end-systolic and end-diastolic frames based on available frames in VolumeTracings.csv.

    Parameters:
        video_name (str): Name of the video file (e.g., 'sample_video.avi').
        video_folder (str): Path to the folder containing video files.
        filelist_path (str): Path to the FileList.csv.
        tracings_path (str): Path to the VolumeTracings.csv.
    """
    # Load the file list and volume tracings
    file_list_df = pd.read_csv(filelist_path)
    volume_tracings_df = pd.read_csv(tracings_path)

    # Remove .avi extension from the video name for consistent comparison
    video_name_no_ext = video_name.replace('.avi', '')

    # Filter the file list for the given video
    video_info = file_list_df[file_list_df['FileName'] == video_name_no_ext]
    if video_info.empty:
        print(f"Video {video_name} not found in FileList.csv.")
        return

    # Access frame height and width properly using .iloc
    frame_height = int(video_info['FrameHeight'].iloc[0])
    frame_width = int(video_info['FrameWidth'].iloc[0])

    # Filter volume tracings for the given video
    video_tracings = volume_tracings_df[volume_tracings_df['FileName'] == video_name]

    # Check if there are tracings available
    if video_tracings.empty:
        print("Segmentation coordinates for the specified video not found.")
        return

    # Identify the unique frames available in tracings (e.g., end-systolic and end-diastolic)
    unique_frames = video_tracings['Frame'].unique()
    if len(unique_frames) < 2:
        print("Insufficient frames for end-systolic and end-diastolic segmentation.")
        return

    # Assume the smallest frame number as end-diastolic and the largest as end-systolic
    end_diastolic_frame_num = unique_frames.min()
    end_systolic_frame_num = unique_frames.max()

    # Extract the tracings for these frames
    end_diastolic_frame = video_tracings[video_tracings['Frame'] == end_diastolic_frame_num]
    end_systolic_frame = video_tracings[video_tracings['Frame'] == end_systolic_frame_num]

    # Create masks for the frames
    def create_mask(tracings, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for _, row in tracings.iterrows():
            x1, y1, x2, y2 = int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2'])
            mask[y1:y2, x1:x2] = 1  # Fill the region in the mask
        return mask

    systolic_mask = create_mask(end_systolic_frame, frame_height, frame_width)
    diastolic_mask = create_mask(end_diastolic_frame, frame_height, frame_width)

    # Plot the masks
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(systolic_mask, cmap='viridis')
    axs[0].set_title(f'End-Systolic Mask (Frame {end_systolic_frame_num})')
    axs[0].axis('off')

    axs[1].imshow(diastolic_mask, cmap='viridis')
    axs[1].set_title(f'End-Diastolic Mask (Frame {end_diastolic_frame_num})')
    axs[1].axis('off')

    plt.show()

# Example call to the function (assuming video is in the specified folder and files are available)
# visualize_segmentation_for_systolic_diastolic("0X100009310A3BD7FC.avi", "/path/to/video_folder", file_list_path, volume_tracings_path)



# Usage
if __name__ == "__main__":
    visualize_segmentation_masks('your_sample_video.avi', 'Videos')
