import base64
import os
import cv2
import numpy as np
import requests
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
REPO_OWNER = "jm1d22"
REPO_NAME = "ColourableZebra3D"
BRANCH = "main"
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
COUNTER_FILE = SCRIPT_DIR / ".texture_counter"
TOKEN_FILE = SCRIPT_DIR / "token.txt"


def load_github_token() -> str:
    """Reads the GitHub Personal Access Token from token.txt in the script directory."""
    if not TOKEN_FILE.exists():
        print(f"[WARNING] '{TOKEN_FILE.name}' not found in {SCRIPT_DIR}.")
        print("[WARNING] Remote GitHub upload will be skipped.")
        return ""
    
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if not token:
                print(f"[WARNING] '{TOKEN_FILE.name}' is empty. Remote GitHub upload will be skipped.")
                return ""
            return token
    except Exception as e:
        print(f"[ERROR] Failed to read '{TOKEN_FILE.name}': {e}")
        return ""


def get_next_texture_filename() -> str:
    """Reads a local counter file to generate sequential texture names (e.g., Texture_0001.jpg)."""
    count = 1
    if COUNTER_FILE.exists():
        try:
            with open(COUNTER_FILE, "r") as f:
                count = int(f.read().strip()) + 1
        except ValueError:
            count = 1

    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))

    return f"Texture_{count:04d}.jpg"


def capture_photo_from_webcam(save_path: Path) -> bool:
    """Opens a live webcam preview window. Press Spacebar to take photo, ESC to quit."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
        cv2.putText(
            preview_frame,
            "Press SPACE to Capture | ESC to Exit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

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


def upload_texture_to_github(
    local_jpg_path: Path,
    remote_filename: str,
    commit_msg: str = "Upload scanned zebra texture",
):
    """Uploads scanned texture directly into docs/assets/textures/ on GitHub via API."""
    token = load_github_token()
    if not token:
        print("[INFO] Skipping auto-upload due to missing/invalid token.")
        return

    remote_path = f"docs/assets/textures/{remote_filename}"
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{remote_path}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    with open(local_jpg_path, "rb") as img_file:
        encoded_content = base64.b64encode(img_file.read()).decode("utf-8")

    get_res = requests.get(url, headers=headers)
    sha = get_res.json().get("sha") if get_res.status_code == 200 else None

    payload = {
        "message": commit_msg,
        "content": encoded_content,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        print(f"[SUCCESS] Texture uploaded to GitHub repository: {remote_path}")
        print(f"Commit URL: {response.json()['commit']['html_url']}")
    else:
        print(f"[ERROR] GitHub upload failed ({response.status_code}): {response.json()}")


def process_zebra_textures():
    assets_image_dir = PROJECT_ROOT / "docs" / "assets" / "image"
    assets_textures_dir = PROJECT_ROOT / "docs" / "assets" / "textures"

    assets_image_dir.mkdir(parents=True, exist_ok=True)
    assets_textures_dir.mkdir(parents=True, exist_ok=True)

    captured_image_path = assets_image_dir / "captured_coloring_sheet.jpg"

    success = capture_photo_from_webcam(captured_image_path)
    if not success:
        return

    sequential_filename = get_next_texture_filename()
    output_2k_path = assets_textures_dir / sequential_filename

    img = cv2.imread(str(captured_image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image from {captured_image_path}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        raise ValueError("No ArUco markers detected. Ensure all 4 tags are visible.")

    ids = ids.flatten()

    target_centers = {
        1: [50.0, 50.0],
        2: [1997.0, 50.0],
        3: [50.0, 1997.0],
        4: [1997.0, 1997.0],
    }

    src_pts = []
    dst_pts = []

    for marker_id in [1, 2, 3, 4]:
        if marker_id not in ids:
            raise ValueError(f"Missing Marker ID {marker_id}. Detected IDs: {list(ids)}")

        idx = np.where(ids == marker_id)[0][0]
        marker_corners = corners[idx][0]

        center_x = float(np.mean(marker_corners[:, 0]))
        center_y = float(np.mean(marker_corners[:, 1]))

        src_pts.append([center_x, center_y])
        dst_pts.append(target_centers[marker_id])

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    texture_2k = cv2.warpPerspective(img, matrix, (2048, 2048), flags=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(output_2k_path), texture_2k, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Local 2K Texture Saved: {output_2k_path}")
    print("=" * 60)

    upload_texture_to_github(
        local_jpg_path=output_2k_path,
        remote_filename=sequential_filename,
        commit_msg=f"Auto-upload scanned texture {sequential_filename}",
    )


if __name__ == "__main__":
    process_zebra_textures()