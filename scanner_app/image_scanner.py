import base64
import os
import re
import cv2
import numpy as np
import requests
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from blender_bake import run_blender_bake
from config import TOKEN_FILE

# ============================================================
# CONFIGURATION
# ============================================================
REPO_OWNER = "jm1d22"
REPO_NAME = "ColourableZebra3D"
BRANCH = "main"
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def prompt_zebra_name() -> str:
    """Displays a GUI text input box for the user to name their zebra."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    name = simpledialog.askstring("Name Your Zebra", "Enter a name for your Zebra:")
    root.destroy()
    
    if not name:
        return "UnnamedZebra"
    
    # Sanitize string for valid filenames
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', name.strip())
    return clean_name if clean_name else "UnnamedZebra"

def load_github_token() -> str:
    if not TOKEN_FILE.exists():
        print(f"[WARNING] '{TOKEN_FILE.name}' not found. Remote upload skipped.")
        return ""
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[ERROR] Failed to read token: {e}")
        return ""

def capture_photo_from_webcam(save_path: Path) -> bool:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return False

    print("\n" + "=" * 60)
    print("LIVE CAMERA FEED OPENED -> Press SPACEBAR to capture | ESC to exit")
    print("=" * 60)

    captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preview_frame = frame.copy()
        cv2.putText(preview_frame, "Press SPACE to Capture | ESC to Exit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture Coloring Sheet", preview_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), frame)
            captured = True
            break
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

def upload_texture_to_github(local_file_path: Path, remote_filename: str):
    token = load_github_token()
    if not token:
        return

    remote_path = f"docs/assets/textures/{remote_filename}"
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{remote_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    with open(local_file_path, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")

    get_res = requests.get(url, headers=headers)
    sha = get_res.json().get("sha") if get_res.status_code == 200 else None

    payload = {
        "message": f"Auto-upload texture {remote_filename}",
        "content": encoded_content,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        print(f"[SUCCESS] Uploaded to GitHub: {remote_path}")
    else:
        print(f"[ERROR] Upload failed ({response.status_code}): {response.json()}")

def process_zebra_textures():
    assets_image_dir = PROJECT_ROOT / "docs" / "assets" / "image"
    assets_textures_dir = PROJECT_ROOT / "docs" / "assets" / "textures"

    assets_image_dir.mkdir(parents=True, exist_ok=True)
    assets_textures_dir.mkdir(parents=True, exist_ok=True)

    captured_image_path = assets_image_dir / "captured_coloring_sheet.jpg"
    if not capture_photo_from_webcam(captured_image_path):
        return

    # Prompt user for Zebra Name
    zebra_name = prompt_zebra_name()
    print(f"\n[INFO] Zebra named: '{zebra_name}'")

    output_lowpoly_jpg = assets_textures_dir / f"{zebra_name}.jpg"
    output_highpoly_png = assets_textures_dir / f"{zebra_name}_HighPoly.png"

    # Homography ArUco Warp
    img = cv2.imread(str(captured_image_path))
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        raise ValueError("No ArUco markers detected.")

    ids = ids.flatten()
    target_centers = {1: [50.0, 50.0], 2: [1997.0, 50.0], 3: [50.0, 1997.0], 4: [1997.0, 1997.0]}
    src_pts, dst_pts = [], []

    for marker_id in [1, 2, 3, 4]:
        if marker_id not in ids:
            raise ValueError(f"Missing Marker ID {marker_id}")
        idx = np.where(ids == marker_id)[0][0]
        mc = corners[idx][0]
        src_pts.append([float(np.mean(mc[:, 0])), float(np.mean(mc[:, 1]))])
        dst_pts.append(target_centers[marker_id])

    matrix, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
    texture_2k = cv2.warpPerspective(img, matrix, (2048, 2048), flags=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(output_lowpoly_jpg), texture_2k, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SUCCESS] Low-Poly Texture Saved: {output_lowpoly_jpg.name}")

    # Run Blender Headless Bake
    print("\n[INFO] Starting Headless Blender Bake for High-Poly texture...")
    run_blender_bake(output_lowpoly_jpg, output_highpoly_png)

    # Upload Both Textures to GitHub
    upload_texture_to_github(output_lowpoly_jpg, f"{zebra_name}.jpg")
    upload_texture_to_github(output_highpoly_png, f"{zebra_name}_HighPoly.png")

if __name__ == "__main__":
    process_zebra_textures()