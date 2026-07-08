import cv2
import mediapipe as mp
import time
import sys
import os
import urllib.request
import zipfile
import shutil
import random

# Add the project root directory to sys.path so we can import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.hand import hand_landmarks
from utils.pose import pose_landmarks
from utils.hand import hand_physics
from utils.fpga.fpga_packet import FPGAPacket

def download_with_progress(url, filepath):
    """Downloads a file from a URL with a console progress bar."""
    print(f"Downloading {url}...")
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1 MB blocks
            downloaded = 0
            with open(filepath, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='\r')
            print()
    except Exception as e:
        print(f"Failed to download: {e}")

def main():
    val_dir = os.path.join(project_root, "testing-validation")
    images_dir = os.path.join(val_dir, "images")
    hands_dir = os.path.join(images_dir, "hands")
    pose_dir = os.path.join(images_dir, "pose")

    # 1. Clear existing images directory and make subfolders
    print("--- 1. Resetting validation images directory ---")
    if os.path.exists(images_dir):
        print("Deleting current images directory content...")
        shutil.rmtree(images_dir)
    
    os.makedirs(hands_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # 2. Download datasets if they are not cached in the validation directory
    hand_zip_path = os.path.join(val_dir, "rps_data_sample.zip")
    pose_zip_path = os.path.join(val_dir, "yoga_poses.zip")

    if not os.path.exists(hand_zip_path):
        print("Hand dataset zip not found. Downloading...")
        download_with_progress(
            "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip",
            hand_zip_path
        )
    else:
        print("Found cached hand dataset zip.")

    if not os.path.exists(pose_zip_path):
        print("Pose dataset zip not found. Downloading...")
        download_with_progress(
            "http://download.tensorflow.org/data/pose_classification/yoga_poses.zip",
            pose_zip_path
        )
    else:
        print("Found cached pose dataset zip.")

    # 3. Extract exactly 10 images for hands and 10 images for pose
    print("\n--- 2. Extracting sample validation images ---")
    
    # Extract 10 random hand images
    if os.path.exists(hand_zip_path):
        print("Extracting 10 random hand images from dataset...")
        with zipfile.ZipFile(hand_zip_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files = [f for f in image_files if '__macosx' not in f.lower()]
            
            # Select 10 random images
            to_extract = random.sample(image_files, 10) if len(image_files) >= 10 else image_files
            
            for idx, f in enumerate(to_extract):
                filename = f"hand_{idx}_{os.path.basename(f)}"
                dest_path = os.path.join(hands_dir, filename)
                with zip_ref.open(f) as src_file, open(dest_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
        print(f"Extracted 10 random images into {hands_dir}")

    # Extract 10 random pose images
    if os.path.exists(pose_zip_path):
        print("Extracting 10 random pose images from dataset...")
        with zipfile.ZipFile(pose_zip_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files = [f for f in image_files if '__macosx' not in f.lower()]
            
            # Select 10 random images
            to_extract = random.sample(image_files, 10) if len(image_files) >= 10 else image_files
            
            for idx, f in enumerate(to_extract):
                filename = f"pose_{idx}_{os.path.basename(f)}"
                dest_path = os.path.join(pose_dir, filename)
                with zip_ref.open(f) as src_file, open(dest_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
        print(f"Extracted 10 random images into {pose_dir}")

    # 4. Initialize landmarkers in IMAGE mode
    print("\n--- 3. Initializing Landmarkers (IMAGE mode) ---")
    hand_model_path = os.path.join(project_root, "models", "hand_landmarker.task")
    pose_model_path = os.path.join(project_root, "models", "pose_landmarker_lite.task")

    landmarker = hand_landmarks.create_hand_landmarker(
        running_mode=hand_landmarks.VisionRunningMode.IMAGE,
        model_path=hand_model_path,
        num_hands=1
    )
    pose_detector = pose_landmarks.create_pose_landmarker(
        running_mode=pose_landmarks.VisionRunningMode.IMAGE,
        model_path=pose_model_path
    )
    fpga_pkt = FPGAPacket()

    # Dictionary to collect all biometric data points
    raw_data = {
        "pinky": [],
        "ring": [],
        "middle": [],
        "index": [],
        "thumb_palm": [],
        "thumb": [],
        "wrist": [],
        "elbow": []
    }

    mapped_data = {
        "pinky": [],
        "ring": [],
        "middle": [],
        "index": [],
        "thumb_palm": [],
        "thumb": [],
        "wrist": [],
        "elbow": []
    }

    # 5. Process Hand images
    print("\n--- 4. Processing hand images ---")
    hand_files = [os.path.join(hands_dir, f) for f in os.listdir(hands_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filepath in hand_files:
        filename = os.path.basename(filepath)
        print(f"Processing hand image: {filename}...")
        frame = cv2.imread(filepath)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            print(f"  Detected {len(result.hand_landmarks)} hand(s)")
            for i, hand_landmarks_list in enumerate(result.hand_landmarks):
                hand_world_landmarks = result.hand_world_landmarks[i]

                # Flexion angles
                pinky_flex  = hand_physics.get_finger_flexion(hand_world_landmarks, 4)
                ring_flex   = hand_physics.get_finger_flexion(hand_world_landmarks, 3)
                middle_flex = hand_physics.get_finger_flexion(hand_world_landmarks, 2)
                index_flex  = hand_physics.get_finger_flexion(hand_world_landmarks, 1)
                thumb_flex  = hand_physics.get_finger_flexion(hand_world_landmarks, 0)

                # Thumb palm (opposition distance in mm)
                thumb_palm = hand_physics.calculate_thumb_opposition(hand_world_landmarks)

                # Wrist yaw
                _, wrist_yaw = hand_physics.calculate_wrist_angles(hand_world_landmarks)

                # Store raw values
                raw_data["pinky"].append(pinky_flex)
                raw_data["ring"].append(ring_flex)
                raw_data["middle"].append(middle_flex)
                raw_data["index"].append(index_flex)
                raw_data["thumb_palm"].append(thumb_palm)
                raw_data["thumb"].append(thumb_flex)
                raw_data["wrist"].append(wrist_yaw)

                # Store mapped/clamped values (as sent in the FPGA packet)
                mapped_data["pinky"].append(fpga_pkt._scale(pinky_flex, fpga_pkt.FLEX_MIN, fpga_pkt.FLEX_MAX))
                mapped_data["ring"].append(fpga_pkt._scale(ring_flex, fpga_pkt.FLEX_MIN, fpga_pkt.FLEX_MAX))
                mapped_data["middle"].append(fpga_pkt._scale(middle_flex, fpga_pkt.FLEX_MIN, fpga_pkt.FLEX_MAX))
                mapped_data["index"].append(fpga_pkt._scale(index_flex, fpga_pkt.FLEX_MIN, fpga_pkt.FLEX_MAX))
                mapped_data["thumb_palm"].append(fpga_pkt._scale(thumb_palm, fpga_pkt.OPP_MIN, fpga_pkt.OPP_MAX, invert=True))
                mapped_data["thumb"].append(fpga_pkt._scale(thumb_flex, fpga_pkt.FLEX_MIN, fpga_pkt.FLEX_MAX))
                mapped_data["wrist"].append(max(0, min(180, 180 - int(wrist_yaw))))
        else:
            print("  No hands detected.")

    # 6. Process Pose images
    print("\n--- 5. Processing pose images ---")
    pose_files = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filepath in pose_files:
        filename = os.path.basename(filepath)
        print(f"Processing pose image: {filename}...")
        frame = cv2.imread(filepath)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_result = pose_detector.detect(mp_image)

        if pose_result.pose_landmarks:
            print("  Detected body pose - extracting elbow angle")
            # Calculate only the right elbow angle to count as 1 sample per image
            elbow = pose_landmarks.get_elbow_angle(pose_result.pose_landmarks[0], "Right")

            raw_data["elbow"].append(elbow)
            mapped_data["elbow"].append(max(0, min(180, 180 - int(elbow))))
        else:
            print("  No body pose detected.")

    # Close landmarkers
    landmarker.close()
    pose_detector.close()

    # 7. Generate report table
    print("\n--- 6. Generating Validation Report ---")
    
    report_lines = []
    report_lines.append("="*95)
    report_lines.append("BIOMETRIC FEATURES VALIDATION REPORT")
    report_lines.append("="*95)
    report_lines.append(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of hand samples processed: {len(raw_data['pinky'])}")
    report_lines.append(f"Number of pose samples processed: {len(raw_data['elbow'])}")
    report_lines.append("-"*95)
    report_lines.append(
        f"{'Biometric Feature':<18} | {'Packet Expected':<15} | {'Actual Mapped (Min - Max)':<25} | {'Actual Raw (Min - Max)':<25}"
    )
    report_lines.append("-"*95)

    for feature in raw_data.keys():
        raw_list = raw_data[feature]
        mapped_list = mapped_data[feature]

        if raw_list:
            min_raw, max_raw = min(raw_list), max(raw_list)
            min_mapped, max_mapped = min(mapped_list), max(mapped_list)
            raw_range_str = f"{min_raw:.1f} - {max_raw:.1f}"
            mapped_range_str = f"{min_mapped} - {max_mapped}"
        else:
            raw_range_str = "N/A"
            mapped_range_str = "N/A"

        report_lines.append(
            f"{feature:<18} | {'0 - 180':<15} | {mapped_range_str:<25} | {raw_range_str:<25}"
        )
    
    report_lines.append("="*95)
    
    # Save to disk
    report_content = "\n".join(report_lines)
    report_path = os.path.join(val_dir, "biometrics_validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(report_content)
    print(f"\nReport successfully saved to: {report_path}")

if __name__ == "__main__":
    main()
