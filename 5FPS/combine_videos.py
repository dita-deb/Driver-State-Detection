import cv2
import os

def generate_processed_video(input_dir, output_video_path, frame_rate=30):
    # Read image files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    
    if not image_files:
        raise ValueError("No images found in the directory.")
    
    # Get dimensions from the first image
    first_image = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width, _ = first_image.shape

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        video_writer.write(image)

    video_writer.release()
    print(f"Processed video saved successfully at {output_video_path}")

# Example usage
input_images_dir = r'C:\Users\mc2di\project_1\Driver-State-Detection\processed_images'
processed_video_path = r'C:\Users\mc2di\project_1\Driver-State-Detection\processed_video.mp4'
generate_processed_video(input_images_dir, processed_video_path)


import cv2
import numpy as np
import os

def combine_videos(video_path_original, video_path_processed, output_combined_video_path):
    # Check if the files exist
    if not os.path.isfile(video_path_original):
        raise ValueError(f"Error: Original video file does not exist at {video_path_original}")
    if not os.path.isfile(video_path_processed):
        raise ValueError(f"Error: Processed video file does not exist at {video_path_processed}")

    # Open the original and processed videos
    cap_original = cv2.VideoCapture(video_path_original)
    cap_processed = cv2.VideoCapture(video_path_processed)

    if not cap_original.isOpened():
        raise ValueError(f"Error: Could not open original video file at {video_path_original}")
    if not cap_processed.isOpened():
        raise ValueError(f"Error: Could not open processed video file at {video_path_processed}")

    # Get video properties
    frame_width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_original.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object for the combined video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'MJPG' if 'XVID' does not work
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (frame_width * 2, frame_height))

    if not out.isOpened():
        raise ValueError(f"Error: Could not open combined video file for writing at {output_combined_video_path}")

    while cap_original.isOpened() and cap_processed.isOpened():
        ret_original, frame_original = cap_original.read()
        ret_processed, frame_processed = cap_processed.read()
        
        if not ret_original or not ret_processed:
            print("Error reading frames from one or both videos.")
            break
        
        # Concatenate images horizontally
        combined_frame = np.hstack((frame_original, frame_processed))
        
        # Write combined frame to video
        out.write(combined_frame)

    # Release video objects
    cap_original.release()
    cap_processed.release()
    out.release()

    print(f"Combined video saved successfully at {output_combined_video_path}")

# Example usage
original_video_path = r'C:\Users\mc2di\project_1\Driver-State-Detection\original_video.mp4'
processed_video_path = r'C:\Users\mc2di\project_1\Driver-State-Detection\processed_video.mp4'
combined_video_path = r'C:\Users\mc2di\project_1\Driver-State-Detection\combined_video.mp4'
combine_videos(original_video_path, processed_video_path, combined_video_path)
