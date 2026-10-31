"""Paper-based reconstruction, not recovered original code or weights."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'yolov13'
if 'ultralytics' in sys.modules:
    loaded = Path(sys.modules['ultralytics'].__file__).resolve()
    if VENDOR not in loaded.parents:
        raise ImportError('Import spectral_yolo before any separately installed ultralytics package.')
sys.path.insert(0, str(VENDOR))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
__version__ = '0.1.0'
