import os
import cv2
import numpy as np
import random

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def create_synthetic_hand_image(label_id, width=640, height=480):
    """
    Generates a synthetic hand-like representation using OpenCV shapes.
    Draws a forearm, palm, and finger shapes with color variations,
    calculates the precise bounding box, and returns the image and bounding box.
    """
    # Create background (cluttered background to simulate robustness)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
    img[:] = bg_color
    
    # Draw some background "clutter" (simulating a room)
    for _ in range(5):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        cv2.rectangle(img, pt1, pt2, color, -1)
        
    # Hand skin color parameters (varying tones)
    skin_b = random.randint(120, 180)
    skin_g = random.randint(150, 210)
    skin_r = random.randint(200, 255)
    skin_color = (skin_b, skin_g, skin_r)
    
    # Draw hand at a random position inside the safe zone
    # Safe zone: [100, 100] to [width-250, height-250] to keep bounding boxes clean
    hand_w = random.randint(120, 180)
    hand_h = random.randint(140, 200)
    x = random.randint(100, width - hand_w - 100)
    y = random.randint(100, height - hand_h - 100)
    
    # Draw forearm
    cv2.rectangle(img, (x + int(hand_w * 0.3), y + hand_h), (x + int(hand_w * 0.7), height), skin_color, -1)
    
    # Draw palm (ellipse or rectangle)
    center = (x + int(hand_w * 0.5), y + int(hand_h * 0.6))
    axes = (int(hand_w * 0.4), int(hand_h * 0.35))
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)
    
    # Draw fingers representing the sign class (varying shapes for A-H)
    # A: closed fist, thumb on side
    # B: open hand, fingers straight up
    # C: curved hand shape
    # D: index pointing up, others closed
    # E: closed fist, fingers curled in tight
    # F: index and thumb touching, others up
    # G: index and thumb pointing sideways (pointing gesture)
    # H: index and middle fingers pointing sideways
    
    if label_id == 0: # Class 'A' (Fist)
        cv2.circle(img, (x + int(hand_w * 0.25), y + int(hand_h * 0.45)), int(hand_w * 0.15), skin_color, -1)
        cv2.circle(img, (x + int(hand_w * 0.5), y + int(hand_h * 0.45)), int(hand_w * 0.15), skin_color, -1)
        cv2.circle(img, (x + int(hand_w * 0.75), y + int(hand_h * 0.45)), int(hand_w * 0.15), skin_color, -1)
        # Thumb folded
        cv2.ellipse(img, (x + int(hand_w * 0.8), y + int(hand_h * 0.65)), (int(hand_w * 0.25), int(hand_h * 0.12)), 45, 0, 360, skin_color, -1)
    elif label_id == 1: # Class 'B' (Open Hand)
        # 4 straight fingers
        for i in range(4):
            fx = x + int(hand_w * (0.2 + i * 0.2))
            cv2.ellipse(img, (fx, y + int(hand_h * 0.15)), (int(hand_w * 0.08), int(hand_h * 0.25)), 0, 0, 360, skin_color, -1)
        # Thumb open
        cv2.ellipse(img, (x + int(hand_w * 0.95), y + int(hand_h * 0.5)), (int(hand_w * 0.25), int(hand_h * 0.08)), -30, 0, 360, skin_color, -1)
    elif label_id == 2: # Class 'C' (Curved)
        cv2.ellipse(img, (x + int(hand_w * 0.5), y + int(hand_h * 0.5)), (int(hand_w * 0.4), int(hand_h * 0.45)), 0, 30, 330, skin_color, 25)
    elif label_id == 3: # Class 'D' (Index Pointing Up)
        # Index straight up
        cv2.ellipse(img, (x + int(hand_w * 0.45), y + int(hand_h * 0.1)), (int(hand_w * 0.08), int(hand_h * 0.3)), 0, 0, 360, skin_color, -1)
        # Others curled
        for fx in [x + int(hand_w * 0.2), x + int(hand_w * 0.65), x + int(hand_w * 0.85)]:
            cv2.circle(img, (fx, y + int(hand_h * 0.45)), int(hand_w * 0.12), skin_color, -1)
    elif label_id == 4: # Class 'E' (Curled fist)
        cv2.circle(img, (x + int(hand_w * 0.5), y + int(hand_h * 0.45)), int(hand_w * 0.35), skin_color, -1)
    elif label_id == 5: # Class 'F' (OK Sign)
        # 3 straight fingers (middle, ring, pinky)
        for i in range(3):
            fx = x + int(hand_w * (0.45 + i * 0.2))
            cv2.ellipse(img, (fx, y + int(hand_h * 0.15)), (int(hand_w * 0.08), int(hand_h * 0.25)), 0, 0, 360, skin_color, -1)
        # Circle between index and thumb
        cv2.circle(img, (x + int(hand_w * 0.25), y + int(hand_h * 0.55)), int(hand_w * 0.18), skin_color, 12)
    elif label_id == 6: # Class 'G' (Pointing sideways)
        # Index pointing left
        cv2.ellipse(img, (x - int(hand_w * 0.1), y + int(hand_h * 0.4)), (int(hand_w * 0.35), int(hand_h * 0.08)), 0, 0, 360, skin_color, -1)
        # Thumb pointing up
        cv2.ellipse(img, (x + int(hand_w * 0.35), y + int(hand_h * 0.2)), (int(hand_w * 0.08), int(hand_h * 0.2)), 0, 0, 360, skin_color, -1)
    else: # Class 'H' (Index and Middle pointing sideways)
        # Index pointing left
        cv2.ellipse(img, (x - int(hand_w * 0.1), y + int(hand_h * 0.35)), (int(hand_w * 0.35), int(hand_h * 0.07)), 0, 0, 360, skin_color, -1)
        # Middle pointing left
        cv2.ellipse(img, (x - int(hand_w * 0.15), y + int(hand_h * 0.5)), (int(hand_w * 0.35), int(hand_h * 0.07)), 0, 0, 360, skin_color, -1)
        
    # Overlay some random shadows/lighting (brightness variation)
    img = img.astype(np.int16)
    noise = np.random.randint(-15, 15, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Calculate bounding box coordinates
    # Let's derive it from our hand coordinates and add a small padding
    pad = 10
    xmin = max(0, x - pad - (int(hand_w * 0.25) if label_id in [6, 7] else 0))
    ymin = max(0, y - pad - (int(hand_h * 0.1) if label_id in [1, 3] else 0))
    xmax = min(width, x + hand_w + pad + (int(hand_w * 0.1) if label_id in [1, 5] else 0))
    ymax = min(height, y + hand_h + pad)
    
    # Convert to YOLO normalized coordinates
    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + box_w / 2.0
    y_center = ymin + box_h / 2.0
    
    yolo_x = x_center / width
    yolo_y = y_center / height
    yolo_w = box_w / width
    yolo_h = box_h / height
    
    return img, (yolo_x, yolo_y, yolo_w, yolo_h)

def generate_synthetic_dataset(output_dir, num_per_class=20):
    """
    Generates a synthetic annotated dataset for ASL detection.
    Splits data into train (80%) and validation (20%) sets.
    """
    print(f"Generating synthetic ASL dataset in: {output_dir}")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    total_images = len(CLASSES) * num_per_class
    image_count = 0
    
    for class_idx, class_name in enumerate(CLASSES):
        for img_num in range(num_per_class):
            # Split: 80% train, 20% validation
            split = 'train' if img_num < int(num_per_class * 0.8) else 'val'
            
            img, bbox = create_synthetic_hand_image(class_idx)
            
            # Save image
            img_filename = f"{class_name}_{img_num}.jpg"
            img_path = os.path.join(output_dir, 'images', split, img_filename)
            cv2.imwrite(img_path, img)
            
            # Save labels
            label_filename = f"{class_name}_{img_num}.txt"
            label_path = os.path.join(output_dir, 'labels', split, label_filename)
            with open(label_path, 'w') as f:
                f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
            image_count += 1
            
    print(f"Generated {image_count} images across {len(CLASSES)} classes.")
    
    # Create dataset yaml file for YOLOv8
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(CLASSES):
        yaml_content += f"  {idx}: {name}\n"
        
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YOLO dataset config written to: {yaml_path}")

def run_webcam_capture(output_dir):
    """
    Optional webcam capture tool for local environments with a camera.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Ensure camera is connected.")
        return
        
    print("Webcam initialized. Press 's' to save a frame, 'c' to change class, 'q' to quit.")
    current_class_idx = 0
    os.makedirs(os.path.join(output_dir, 'captured'), exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Draw target capture box
        box_size = 200
        xmin, ymin = (w - box_size) // 2, (h - box_size) // 2
        xmax, ymax = xmin + box_size, ymin + box_size
        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Class: {CLASSES[current_class_idx]} (Press 'c' to change)", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Webcam Capture Tool", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_class_idx = (current_class_idx + 1) % len(CLASSES)
        elif key == ord('s'):
            # Save image and calculate bounding box
            img_id = len(os.listdir(os.path.join(output_dir, 'captured'))) // 2
            img_name = f"{CLASSES[current_class_idx]}_cap_{img_id}.jpg"
            cv2.imwrite(os.path.join(output_dir, 'captured', img_name), frame)
            
            # Label
            label_name = f"{CLASSES[current_class_idx]}_cap_{img_id}.txt"
            x_center = (xmin + box_size / 2.0) / w
            y_center = (ymin + box_size / 2.0) / h
            yolo_w = box_size / w
            yolo_h = box_size / h
            
            with open(os.path.join(output_dir, 'captured', label_name), 'w') as f:
                f.write(f"{current_class_idx} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                
            print(f"Saved captured frame for Class {CLASSES[current_class_idx]}.")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ASL Dataset Creator (Synthetic / Capture)")
    parser.add_argument('--mode', type=str, default='synthetic', choices=['synthetic', 'capture'],
                        help="Choose 'synthetic' to generate a mock dataset, or 'capture' to capture from webcam.")
    parser.add_argument('--output', type=str, default='asl_detection/data/dataset', help="Dataset directory path")
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        generate_synthetic_dataset(args.output)
    else:
        run_webcam_capture(args.output)
