#!/usr/bin/env python3
import os
import glob
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils import detect_crop_params, apply_crop

def process_image(path, crop_params):
    """
    Load with OpenCV, apply the RGB-based crop, then save back in BGR.
    """
    # Read BGR, convert to RGB for cropping
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"⚠️  Failed to read {path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Apply crop (utils.apply_crop expects RGB numpy array)
    cropped_rgb = apply_crop(rgb, crop_params)
    
    # Convert back to BGR and overwrite
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, cropped_bgr)
    print(f"✅  Cropped: {os.path.basename(path)}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch-crop fisheye images (OpenCV + threads)."
    )
    parser.add_argument(
        "folder",
        help="Directory containing your images (*.jpg, *.jpeg, *.png)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker threads (default: CPU count)"
    )
    args = parser.parse_args()
    folder = args.folder

    # collect images
    exts = ("*.jpg","*.jpeg","*.png")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(folder, ext)))
    img_paths = sorted(img_paths)
    if not img_paths:
        print(f"No images found in {folder}")
        return

    # detect crop params from first image
    first = img_paths[0]
    print(f"▶️  Detecting crop params from: {os.path.basename(first)}")
    bgr0 = cv2.imread(first, cv2.IMREAD_COLOR)
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
    crop_params = detect_crop_params(rgb0)
    print("Detected crop params:", crop_params)

    # parallel processing
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        # pass both path and params to worker
        futures = [
            pool.submit(process_image, path, crop_params)
            for path in img_paths
        ]
        for f in futures:
            f.result()  # wait and propagate errors

if __name__ == "__main__":
    main()
