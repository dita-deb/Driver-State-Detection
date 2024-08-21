import cv2
import os

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Read and save frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or error reading frame.")
            break
        # Construct filename
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        # Save the frame
        if cv2.imwrite(frame_filename, frame):
            print(f"Saved frame {frame_count} to {frame_filename}")
        else:
            print(f"Failed to save frame {frame_count} to {frame_filename}")
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")

# Paths
video_path = "C:\\Users\\mc2di\\project_1\\Driver-State-Detection\\MOVC0008.avi"
output_dir = "C:\\Users\\mc2di\\project_1\\Driver-State-Detection\\output_images"

# Extract frames
extract_frames(video_path, output_dir)
