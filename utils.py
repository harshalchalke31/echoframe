import cv2
import pandas as pd
import matplotlib.pyplot as plt


def visualize_segmentation_masks(video_name, video_folder, filelist_path='Filelist.csv', tracings_path='VolumeTracings.csv', num_frames=10):
    """
    Visualize segmentation masks for a specified number of frames from a video in the EchoNet-Pediatric dataset.
    
    Parameters:
        video_name (str): Name of the video file (e.g., 'your_sample_video.avi').
        video_folder (str): Path to the folder containing video files.
        filelist_path (str): Path to 'Filelist.csv'.
        tracings_path (str): Path to 'VolumeTracings.csv'.
        num_frames (int): Number of frames to visualize with segmentation masks.
    """
    # Load the file lists
    filelist_df = pd.read_csv(filelist_path)
    tracings_df = pd.read_csv(tracings_path)

    # Select tracings for the sample video
    sample_tracings = tracings_df[tracings_df['FileName'] == video_name]

    # Load the video
    video_path = f"{video_folder}/{video_name}"
    cap = cv2.VideoCapture(video_path)

    # Select frames to visualize
    frame_indices = sample_tracings['Frame'].unique()[:num_frames]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    frame_count = 0  # Counter for the valid frames to display

    for i, frame_idx in enumerate(frame_indices):
        # Ensure frame_idx is an integer
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
        axes[frame_count].imshow(frame_rgb)
        
        # Overlay segmentation mask
        axes[frame_count].plot(x_coords, y_coords, 'r-', linewidth=2)
        axes[frame_count].set_title(f"Frame {frame_idx}")
        axes[frame_count].axis('off')
        
        frame_count += 1  # Move to the next subplot

        # Break if we have filled all subplots
        if frame_count >= len(axes):
            break

    cap.release()
    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    visualize_segmentation_masks('your_sample_video.avi', 'Videos')
