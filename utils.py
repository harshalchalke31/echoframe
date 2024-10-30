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

# Usage
if __name__ == "__main__":
    visualize_segmentation_masks('your_sample_video.avi', 'Videos')
