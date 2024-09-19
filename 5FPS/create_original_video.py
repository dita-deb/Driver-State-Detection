import cv2
import os

# Path to original images folder and output video
input_folder = r'C:\Users\mc2di\project_1\Driver-State-Detection\output_images\Ramesh_Video_to_Images_5_FPS'
output_video_path_original = r'C:\Users\mc2di\project_1\Driver-State-Detection\original_video.mp4'

# Parameters
frame_rate = 5  # As images were extracted at 5 FPS
frame_size = None  # This will be determined from the images

def load_images_from_folder(folder):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load original images
original_images = load_images_from_folder(input_folder)

# Ensure images are loaded
if not original_images:
    raise ValueError("No images found in the input folder.")

# Determine frame size from the images
height, width, _ = original_images[0].shape
frame_size = (width, height)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'MJPG' if 'XVID' does not work
out = cv2.VideoWriter(output_video_path_original, fourcc, frame_rate, frame_size)

if not out.isOpened():
    raise ValueError("Error: Could not open video file for writing.")

# Write frames to video
for img in original_images:
    out.write(img)

# Release video writer
out.release()

print(f"Original video saved successfully at {output_video_path_original}")
