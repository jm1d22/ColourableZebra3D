import cv2
import numpy as np
from pathlib import Path

def capture_photo_from_webcam(save_path: Path) -> bool:
    """Opens a live webcam preview window. Press Spacebar to take photo, ESC to quit."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 for primary camera
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return False

    print("\n" + "=" * 60)
    print("LIVE CAMERA FEED OPENED")
    print(" -> Press SPACEBAR to capture the picture")
    print(" -> Press ESC to exit")
    print("=" * 60)

    captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab webcam frame.")
            break

        preview_frame = frame.copy()
        cv2.putText(preview_frame, "Press SPACE to Capture | ESC to Exit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Capture Coloring Sheet", preview_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), frame)
            print(f"\n[OK] Photo captured and saved to: {save_path}")
            captured = True
            break
        elif key == 27:  # ESC
            print("\n[CANCELLED] Camera capture aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

def process_zebra_textures():
    # 1. Resolve local directories relative to script location
    script_dir = Path(__file__).resolve().parent
    main_dir = script_dir.parent
    
    assets_image_dir = main_dir / "assets" / "image"
    assets_textures_dir = main_dir / "assets" / "textures"
    
    assets_image_dir.mkdir(parents=True, exist_ok=True)
    assets_textures_dir.mkdir(parents=True, exist_ok=True)

    captured_image_path = assets_image_dir / "captured_coloring_sheet.jpg"

    # 2. Open camera pop-up to take picture
    success = capture_photo_from_webcam(captured_image_path)
    if not success:
        return

    # 3. Define 2K texture output path
    output_2k_path = assets_textures_dir / "Zebra2K.jpg"

    # 4. Read captured photo
    img = cv2.imread(str(captured_image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image from {captured_image_path}")

    # 5. Setup Original ArUco Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 6. Detect markers
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        raise ValueError("No ArUco markers detected in the captured photo. Make sure all 4 tags are visible.")

    ids = ids.flatten()

    # 7. Map center coordinates for 101x101px tags on a 2048x2048 square
    target_centers = {
        1: [50.0, 50.0],       # Top-Left
        2: [1997.0, 50.0],     # Top-Right
        3: [50.0, 1997.0],     # Bottom-Left
        4: [1997.0, 1997.0]    # Bottom-Right
    }

    src_pts = []
    dst_pts = []

    for marker_id in [1, 2, 3, 4]:
        if marker_id not in ids:
            raise ValueError(f"Missing Marker ID {marker_id} in photo. Detected IDs: {list(ids)}")

        idx = np.where(ids == marker_id)[0][0]
        marker_corners = corners[idx][0]

        center_x = float(np.mean(marker_corners[:, 0]))
        center_y = float(np.mean(marker_corners[:, 1]))

        src_pts.append([center_x, center_y])
        dst_pts.append(target_centers[marker_id])

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    # 8. Unwarp Perspective to 2048x2048 (2K Texture)
    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    texture_2k = cv2.warpPerspective(img, matrix, (2048, 2048), flags=cv2.INTER_LANCZOS4)

    # 9. Save 2K texture file to assets/textures/
    cv2.imwrite(str(output_2k_path), texture_2k, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("\n" + "=" * 60)
    print(f"[SUCCESS] 2K Texture Saved: {output_2k_path}")
    print("=" * 60)

if __name__ == "__main__":
    process_zebra_textures()