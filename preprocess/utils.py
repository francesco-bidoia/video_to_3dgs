"""
Module: utils
Contains utility functions for file operations and helper routines.
"""

import os
import shutil

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

def detect_crop_params(image_np, smoothing_window=10, gauss_sigma=5, rise_threshold=2, window_size=10):
    """
    Detects the optimal crop box for a fisheye/180° image by finding the radial brightness
    transition from black border to valid content along each corner, then computing a centered
    crop that preserves the original aspect ratio.

    Args:
        image_np (np.ndarray): Input image as an H×W×3 RGB numpy array.
        smoothing_window (int): Window size for initial moving-average smoothing.
        gauss_sigma (float): Sigma for Gaussian smoothing applied after moving-average.
        rise_threshold (float): Minimum derivative to consider as "start of brightness rise".
        window_size (int): Number of consecutive points in derivative to confirm rise.

    Returns:
        dict: Crop parameters {'left', 'top', 'right', 'bottom'}.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    cx, cy = W // 2, H // 2
    corners = [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]

    def smooth(arr):
        return np.convolve(arr, np.ones(smoothing_window)/smoothing_window, mode='valid')

    def detect_transition(corner):
        steps = int(np.hypot(cx - corner[0], cy - corner[1]))
        xs = np.linspace(cx, corner[0], steps).astype(int)
        ys = np.linspace(cy, corner[1], steps).astype(int)
        profile = gray[ys, xs]
        mov = smooth(profile)
        gau = gaussian_filter1d(mov, sigma=gauss_sigma)
        rev = gau[::-1]
        der = np.diff(rev)

        # Find first sustained positive derivative window
        for i in range(len(der) - window_size):
            if np.all(der[i:i + window_size] > rise_threshold):
                return len(rev) - (i + window_size)
        return None

    # Detect radial distances for all corners
    dists = [detect_transition(c) for c in corners]
    dists = [d for d in dists if d is not None]
    if not dists:
        raise RuntimeError("Failed to detect any transition point; check parameters or image quality.")

    R = int(np.median(dists))

    # Compute centered crop preserving aspect ratio
    ar = W / H
    half_h = R / np.sqrt(ar**2 + 1)
    half_w = ar * half_h

    left = max(0, int(cx - half_w))
    right = min(W, int(cx + half_w))
    top = max(0, int(cy - half_h))
    bottom = min(H, int(cy + half_h))

    return {'left': left, 'top': top, 'right': right, 'bottom': bottom}


def apply_crop(image_np, crop_params):
    """
    Applies the crop to the image using previously computed parameters.

    Args:
        image_np (np.ndarray): Input image as an H×W×C numpy array.
        crop_params (dict): {'left', 'top', 'right', 'bottom'}.

    Returns:
        np.ndarray: Cropped image.
    """
    l, t, r, b = (crop_params['left'], crop_params['top'],
                  crop_params['right'], crop_params['bottom'])
    return image_np[t:b, l:r]


def make_folders(source_path):
    """
    Create required folder structure for the project.
    
    Parameters:
        source_path (str): Base directory path.
    """
    sparse_path = os.path.join(source_path, "sparse/0")
    undistorted_path = os.path.join(source_path, "undistorted")

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(undistorted_path, exist_ok=True)

def clean_paths(source_path, video_n=None, db_path=None, remove_tmp=True):
    """
    Clean the source directory by removing temporary and unwanted folders.
    
    Parameters:
        source_path (str): Base directory path.
        video_n (str): Unused legacy argument kept for backward compatibility.
        db_path (str): Path to the database file to be removed.
    """
    if db_path and os.path.isfile(db_path):
        os.remove(db_path)

    removable_files = [
        os.path.join(source_path, "database.db"),
        os.path.join(source_path, "database_final.db"),
    ]
    for fpath in removable_files:
        if os.path.isfile(fpath):
            os.remove(fpath)

    removable_dirs = [
        "sparse",
        "undistorted",
        "model",
        "d_images",
        "distorted",
        "input",
    ]
    if remove_tmp:
        removable_dirs.insert(0, "tmp")
    removed = []
    for dname in removable_dirs:
        dpath = os.path.join(source_path, dname)
        if os.path.isdir(dpath):
            shutil.rmtree(dpath)
            removed.append(dname)

    if removed:
        print(f"Clean start. removed generated folders: {removed}")

def _name_to_ind(name):
    """
    Convert an image filename to an index.
    
    Parameters:
        name (str): Image filename (e.g., "00000001.jpeg").
        
    Returns:
        int: Numeric index extracted from the filename.
    """
    ind = int(name.split('.')[0])
    return ind
