"""
Module: visibility_prediction
Contains functions for predicting which 3D points are visible from a given camera pose.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import pycolmap
from scipy.spatial import KDTree
from tqdm import tqdm

# Type aliases for clarity
Position = np.ndarray  # 3D position vector
Quaternion = np.ndarray  # Quaternion for rotation (x, y, z, w)
CameraPose = Tuple[Position, Quaternion]  # Combined position and orientation
FrameIndex = int  # Frame index in the video sequence
Point3DID = int  # ID of a 3D point in the reconstruction

def world_to_camera(points: np.ndarray, pose: CameraPose) -> np.ndarray:
    """
    Transform 3D points from world coordinates to camera coordinates.
    
    Parameters:
        points: Array of 3D points in world coordinates (shape: Nx3)
        pose: Camera pose as (position, quaternion)
        
    Returns:
        Array of 3D points in camera coordinates (shape: Nx3)
    """
    from scipy.spatial.transform import Rotation
    
    position, quaternion = pose
    rotation = Rotation.from_quat(quaternion)
    
    # Create rotation matrix
    R = rotation.as_matrix()
    
    # Transform points
    points_cam = np.zeros_like(points)
    for i in range(len(points)):
        # Translate point to camera origin
        translated = points[i] - position
        # Rotate point to camera orientation
        points_cam[i] = R @ translated
    
    return points_cam

def is_point_in_frustum(point_cam: np.ndarray, camera: pycolmap.Camera, 
                        min_depth: float = 0.1, max_depth: float = 100.0,
                        margin: float = 0.0) -> bool:
    """
    Check if a point in camera coordinates is within the camera frustum.
    
    Parameters:
        point_cam: 3D point in camera coordinates
        camera: COLMAP camera object with intrinsic parameters
        min_depth: Minimum depth for visibility
        max_depth: Maximum depth for visibility
        margin: Margin to expand the frustum (in normalized coordinates, 0.0-1.0)
        
    Returns:
        True if the point is within the frustum, False otherwise
    """
    # Check depth bounds
    z = point_cam[2]
    if z < min_depth or z > max_depth:
        return False
    
    # Project point to image coordinates
    x, y = point_cam[0], point_cam[1]
    
    # Simple pinhole projection (adjust based on camera model)
    if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        fx = camera.focal_length_x
        cx, cy = camera.principal_point
        u = fx * x / z + cx
        v = fx * y / z + cy
    elif camera.model == pycolmap.CameraModelId.PINHOLE:
        fx, fy = camera.focal_length_x, camera.focal_length_y
        cx, cy = camera.principal_point
        u = fx * x / z + cx
        v = fy * y / z + cy
    elif camera.model == pycolmap.CameraModelId.OPENCV:
        fx, fy = camera.focal_length_x, camera.focal_length_y
        cx, cy = camera.principal_point_x, camera.principal_point_y
        u = fx * x / z + cx
        v = fy * y / z + cy
    else:
        # For other camera models, use the COLMAP camera.world_to_image function
        # This is a simplification - in practice, you'd need to handle different camera models
        u, v = camera.world_to_image(point_cam)
    
    # Check if the projected point is within the image bounds (with margin)
    width, height = camera.width, camera.height
    min_u = -margin * width
    max_u = (1 + margin) * width
    min_v = -margin * height
    max_v = (1 + margin) * height
    
    return min_u <= u <= max_u and min_v <= v <= max_v

def compute_visibility_for_pose(reconstruction: pycolmap.Reconstruction, 
                               pose: CameraPose,
                               camera: pycolmap.Camera,
                               min_depth: float = 0.1,
                               max_depth: float = 100.0,
                               margin: float = 0.1,
                               handle_occlusion: bool = True,
                               max_occlusion_distance: float = 0.1) -> Dict[Point3DID, float]:
    """
    Compute visibility scores for all 3D points from a given camera pose.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        pose: Camera pose as (position, quaternion)
        camera: COLMAP camera object with intrinsic parameters
        min_depth: Minimum depth for visibility
        max_depth: Maximum depth for visibility
        margin: Margin to expand the frustum (in normalized coordinates)
        handle_occlusion: Whether to handle occlusion
        max_occlusion_distance: Maximum distance for occlusion handling
        
    Returns:
        Dictionary mapping point IDs to visibility scores (0.0-1.0)
    """
    # Extract 3D points from reconstruction
    points3D = reconstruction.points3D
    point_ids = list(points3D.keys())
    point_positions = np.array([points3D[pid].xyz for pid in point_ids])
    
    # Transform points to camera coordinates
    points_cam = world_to_camera(point_positions, pose)
    
    # Check which points are in the frustum
    in_frustum = np.array([
        is_point_in_frustum(p, camera, min_depth, max_depth, margin)
        for p in points_cam
    ])
    
    # Initialize visibility scores
    visibility_scores = {}
    
    # If no occlusion handling, simply return binary visibility
    if not handle_occlusion:
        for i, pid in enumerate(point_ids):
            visibility_scores[pid] = 1.0 if in_frustum[i] else 0.0
        return visibility_scores
    
    # Handle occlusion using z-buffer approach
    # Group points by their projected pixel coordinates
    pixel_to_points = {}
    for i, pid in enumerate(point_ids):
        if in_frustum[i]:
            # Project point to image coordinates (simplified)
            x, y, z = points_cam[i]
            if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                fx = camera.focal_length_x
                cx, cy = camera.principal_point
                u = fx * x / z + cx
                v = fx * y / z + cy
            elif camera.model == pycolmap.CameraModelId.PINHOLE:
                fx, fy = camera.focal_length_x, camera.focal_length_y
                cx, cy = camera.principal_point
                u = fx * x / z + cx
                v = fy * y / z + cy
            elif camera.model == pycolmap.CameraModelId.OPENCV:
                fx, fy = camera.focal_length_x, camera.focal_length_y
                cx, cy = camera.principal_point_x, camera.principal_point_y
                u = fx * x / z + cx
                v = fy * y / z + cy
            else:
                # For other camera models, use the COLMAP camera.world_to_image function
                # This is a simplification - in practice, you'd need to handle different camera models
                u, v = camera.world_to_image(point_cam)
            
            pixel = (u, v)
            if pixel not in pixel_to_points:
                pixel_to_points[pixel] = []
            pixel_to_points[pixel].append((pid, z))
    
    # For each pixel, keep only the closest point (or points within max_occlusion_distance)
    for pixel, points_at_pixel in pixel_to_points.items():
        # Sort points by depth
        points_at_pixel.sort(key=lambda x: x[1])
        
        # The closest point is fully visible
        closest_pid, closest_z = points_at_pixel[0]
        visibility_scores[closest_pid] = 1.0
        
        # Other points at this pixel are occluded based on their distance to the closest point
        for pid, z in points_at_pixel[1:]:
            depth_diff = z - closest_z
            if depth_diff <= max_occlusion_distance:
                # Points very close to the closest point are partially visible
                visibility_scores[pid] = 1.0 - (depth_diff / max_occlusion_distance)
            else:
                # Points far behind the closest point are fully occluded
                visibility_scores[pid] = 0.0
    
    # Set visibility to 0 for points outside the frustum
    for i, pid in enumerate(point_ids):
        if not in_frustum[i] and pid not in visibility_scores:
            visibility_scores[pid] = 0.0
    
    return visibility_scores

def compute_visibility_for_all_poses(reconstruction: pycolmap.Reconstruction,
                                    all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                                    camera: Optional[pycolmap.Camera] = None) -> Dict[FrameIndex, Dict[Point3DID, float]]:
    """
    Compute visibility scores for all 3D points from all camera poses.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        all_poses: Dictionary mapping frame indices to (camera_pose, confidence)
        camera: COLMAP camera object with intrinsic parameters (if None, use the first camera in the reconstruction)
        
    Returns:
        Dictionary mapping frame indices to dictionaries of point visibility scores
    """
    # If no camera is provided, use the first camera in the reconstruction
    if camera is None:
        camera_id = next(iter(reconstruction.cameras))
        camera = reconstruction.cameras[camera_id]
    
    # Compute visibility for each pose
    all_visibility = {}
    for frame_idx, (pose, confidence) in all_poses.items():
        visibility = compute_visibility_for_pose(reconstruction, pose, camera)
        all_visibility[frame_idx] = visibility
        
        # Print progress
        if frame_idx % 100 == 0:
            print(f"Computed visibility for frame {frame_idx} / {len(all_poses.items())}")
    
    return all_visibility

import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_visibility_sampled(
    reconstruction: pycolmap.Reconstruction,
    all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
    camera: Optional[pycolmap.Camera] = None,
    max_compute: int = 1000
) -> Dict[FrameIndex, Dict[Point3DID, float]]:
    """
    Sample at most `max_compute` frames (uniformly),
    compute full visibility on those, then interpolate
    the rest linearly in frame‐index space.
    """

    # 1) Sort your frames and bail out if it's small enough
    frames = sorted(all_poses.keys())
    N = len(frames)
    if N <= max_compute:
        # simply compute all
        return compute_visibility_for_all_poses(reconstruction, all_poses, camera)
    
    # 2) Pick uniform sample positions in [0, N-1]
    sample_positions = np.linspace(0, N-1, max_compute, dtype=int)
    sample_frames    = [frames[p] for p in sample_positions]

    # 3) Compute visibility on the sampled frames (optionally in parallel)
    sampled_vis = {}

    for f in tqdm(sample_frames):
        sampled_vis[f] = compute_visibility_for_pose(
            reconstruction, all_poses[f][0],
            camera if camera else next(iter(reconstruction.cameras.values()))
        )

    # 4) Build the full visibility dict by interpolation
    full_vis = {}
    # Precompute mapping from frame→its index in `frames`
    frame_to_idx = {f:i for i,f in enumerate(frames)}

    for i, f in tqdm(enumerate(frames)):
        # if we sampled it, just copy
        if f in sampled_vis:
            full_vis[f] = sampled_vis[f]
            continue

        # find insertion point in sample_positions
        pos = i
        j = bisect.bisect_left(sample_positions, pos)

        # clamp to edges
        if j == 0:
            full_vis[f] = sampled_vis[frames[sample_positions[0]]]
            continue
        if j == len(sample_positions):
            full_vis[f] = sampled_vis[frames[sample_positions[-1]]]
            continue

        lo_p, hi_p = sample_positions[j-1], sample_positions[j]
        f_lo, f_hi = frames[lo_p], frames[hi_p]
        w = (pos - lo_p) / (hi_p - lo_p)

        vis_lo = sampled_vis[f_lo]
        vis_hi = sampled_vis[f_hi]

        # interpolate every 3D point ID in the model
        this_vis = {}
        for pid in reconstruction.points3D.keys():
            v0 = vis_lo.get(pid, 0.0)
            v1 = vis_hi.get(pid, 0.0)
            this_vis[pid] = (1.0 - w) * v0 + w * v1

        full_vis[f] = this_vis

    return full_vis

def compute_point_observation_potential(reconstruction: pycolmap.Reconstruction,
                                       all_visibility: Dict[FrameIndex, Dict[Point3DID, float]]) -> Dict[FrameIndex, int]:
    """
    Compute the potential number of new 3D point observations for each frame.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        all_visibility: Dictionary mapping frame indices to dictionaries of point visibility scores
        
    Returns:
        Dictionary mapping frame indices to potential observation counts
    """
    # Get currently registered frames
    registered_frames = set()
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        from utils import _name_to_ind
        frame_idx = _name_to_ind(image.name)
        registered_frames.add(frame_idx)
    
    # Get currently observed 3D points for each registered frame
    observed_points = {frame_idx: set() for frame_idx in registered_frames}
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        
        # Get point observations
        for point2D in image.points2D:
            if point2D.has_point3D():
                observed_points[frame_idx].add(point2D.point3D_id)
    
    # Compute potential new observations for each frame
    potential_observations = {}
    for frame_idx, visibility in all_visibility.items():
        if frame_idx in registered_frames:
            # Skip already registered frames
            potential_observations[frame_idx] = 0
            continue
        
        # Count points that are visible from this frame but not yet observed
        visible_points = {pid for pid, score in visibility.items() if score > 0.5}
        already_observed = set()
        for reg_frame in registered_frames:
            already_observed.update(observed_points[reg_frame])
        
        new_observations = visible_points - already_observed
        potential_observations[frame_idx] = len(new_observations)
    
    return potential_observations

def rank_frames_by_information_gain(reconstruction: pycolmap.Reconstruction,
                                   all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                                   all_visibility: Dict[FrameIndex, Dict[Point3DID, float]],
                                   top_k: int = 10) -> List[Tuple[FrameIndex, float]]:
    """
    Rank frames by their potential information gain for the reconstruction.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        all_poses: Dictionary mapping frame indices to (camera_pose, confidence)
        all_visibility: Dictionary mapping frame indices to dictionaries of point visibility scores
        top_k: Number of top frames to return
        
    Returns:
        List of (frame_index, score) tuples, sorted by descending score
    """
    # Get registered frames
    registered_frames = set()
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        from utils import _name_to_ind
        frame_idx = _name_to_ind(image.name)
        registered_frames.add(frame_idx)
    
    # Compute potential observations
    potential_observations = compute_point_observation_potential(reconstruction, all_visibility)
    
    # Compute scores based on potential observations and pose confidence
    scores = {}
    for frame_idx, potential in potential_observations.items():
        if frame_idx in registered_frames:
            continue
        
        pose, confidence = all_poses[frame_idx]
        # Score is a combination of potential observations and pose confidence
        scores[frame_idx] = potential * confidence
    
    # Sort frames by score
    ranked_frames = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top-k frames
    return ranked_frames[:top_k]

def select_frames_for_reconstruction(reconstruction: pycolmap.Reconstruction,
                                    all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                                    n_frames: int = 50,
                                    random_ratio: float = 0.2) -> List[FrameIndex]:
    """
    Select frames for reconstruction based on information gain and random sampling.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        all_poses: Dictionary mapping frame indices to (camera_pose, confidence)
        n_frames: Total number of frames to select
        random_ratio: Ratio of frames to select randomly
        
    Returns:
        List of selected frame indices
    """
    # Compute visibility for all poses
    camera_id = next(iter(reconstruction.cameras))
    camera = reconstruction.cameras[camera_id]
    # all_visibility = compute_visibility_for_all_poses(reconstruction, all_poses, camera)
    all_visibility = compute_visibility_sampled(reconstruction, all_poses, camera)
    
    # Get registered frames
    registered_frames = set()
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        from utils import _name_to_ind
        frame_idx = _name_to_ind(image.name)
        registered_frames.add(frame_idx)
    
    # Determine number of frames to select based on information gain vs. random
    n_info_frames = int(n_frames * (1 - random_ratio))
    n_random_frames = n_frames - n_info_frames
    
    # Select frames based on information gain
    ranked_frames = rank_frames_by_information_gain(reconstruction, all_poses, all_visibility, top_k=n_info_frames)
    selected_frames = [frame_idx for frame_idx, _ in ranked_frames]
    
    # Select random frames from the remaining frames
    remaining_frames = [idx for idx in all_poses.keys() 
                       if idx not in registered_frames and idx not in selected_frames]
    
    if remaining_frames and n_random_frames > 0:
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(len(remaining_frames), 
                                         min(n_random_frames, len(remaining_frames)), 
                                         replace=False)
        random_frames = [remaining_frames[i] for i in random_indices]
        selected_frames.extend(random_frames)
    
    return selected_frames