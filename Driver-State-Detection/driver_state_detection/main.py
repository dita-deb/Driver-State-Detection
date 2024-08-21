import os
import time
import pprint
import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters

def main():

    args = get_args()

    # Set the input and output directories
    args.input_dir = r"C:\Users\mc2di\project_1\Driver-State-Detection\output_images"
    args.output_dir = r"C:\Users\mc2di\project_1\Driver-State-Detection\processed_images"

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )

    prev_time = time.perf_counter()
    fps = 0.0  # Initial FPS value

    t_now = time.perf_counter()

    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    # Read images from the specified directory
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.png'))])

    if not image_files:
        print(f"No images found in directory: {args.input_dir}")
        return

    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error loading image: {image_path}")
            continue

        t_now = time.perf_counter()

        # Initialize the tick counter for processing time measurement
        e1 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        lms = Detector.process(gray).multi_face_landmarks

        if lms:
            landmarks = get_landmarks(lms)

            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
            tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            if frame_det is not None:
                frame = frame_det

            if ear is not None:
                cv2.putText(
                    frame,
                    "EAR:" + str(round(ear, 3)),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if gaze is not None:
                cv2.putText(
                    frame,
                    "Gaze Score:" + str(round(gaze, 3)),
                    (10, 80),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "PERCLOS:" + str(round(perclos_score, 3)),
                (10, 110),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if roll is not None:
                cv2.putText(
                    frame,
                    "roll:" + str(roll.round(1)[0]),
                    (450, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if pitch is not None:
                cv2.putText(
                    frame,
                    "pitch:" + str(pitch.round(1)[0]),
                    (450, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if yaw is not None:
                cv2.putText(
                    frame,
                    "yaw:" + str(yaw.round(1)[0]),
                    (450, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if tired:
                cv2.putText(
                    frame,
                    "TIRED!",
                    (10, 280),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if asleep:
                cv2.putText(
                    frame,
                    "ASLEEP!",
                    (10, 300),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if looking_away:
                cv2.putText(
                    frame,
                    "LOOKING AWAY!",
                    (10, 320),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if distracted:
                cv2.putText(
                    frame,
                    "DISTRACTED!",
                    (10, 340),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        if args.show_fps:
            cv2.putText(
                frame,
                "FPS:" + str(round(fps)),
                (10, 400),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )
        if args.show_proc_time:
            cv2.putText(
                frame,
                "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms",
                (10, 430),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )

        # Save the processed image
        output_path = os.path.join(args.output_dir, image_file)
        cv2.imwrite(output_path, frame)

        # Optionally display the processed frame
        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
