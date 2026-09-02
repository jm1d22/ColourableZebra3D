import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = SCRIPT_DIR / "settings.json"


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_settings() -> dict[str, Path]:
    with SETTINGS_FILE.open(encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    required_keys = ("blender_executable", "blend_file", "token_file")
    missing_keys = [key for key in required_keys if not settings.get(key)]
    if missing_keys:
        raise ValueError(f"Missing setting(s): {', '.join(missing_keys)}")

    return {key: _resolve_path(settings[key]) for key in required_keys}


SETTINGS = load_settings()
BLENDER_EXE = SETTINGS["blender_executable"]
BLEND_FILE = SETTINGS["blend_file"]
TOKEN_FILE = SETTINGS["token_file"]
