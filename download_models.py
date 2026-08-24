"""Download the OpenCV DNN face-detector model files into models/."""
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[skip]     {filename} already present")
            continue
        print(f"[download] {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[done]     saved to {dest}")


if __name__ == "__main__":
    download()
    print("\nAll models ready. Run: python main.py")
