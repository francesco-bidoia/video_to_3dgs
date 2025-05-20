"""
Module: pose_interpolation
Contains functions for interpolating camera poses between keyframes in a video sequence.
"""
import copy
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation, Slerp
import pycolmap
import os
from typing import Dict, List, Tuple, Optional

# Type aliases for clarity
Position = np.ndarray  # 3D position vector
Quaternion = np.ndarray  # Quaternion for rotation (x, y, z, w)
CameraPose = Tuple[Position, Quaternion]  # Combined position and orientation
FrameIndex = int  # Frame index in the video sequence


def extract_camera_trajectory(reconstruction: pycolmap.Reconstruction) -> Dict[FrameIndex, CameraPose]:
    """
    Extract camera poses from a COLMAP reconstruction, sorted by frame index.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        
    Returns:
        Dictionary mapping frame indices to camera poses (position, quaternion)
    """
    from utils import _name_to_ind
    
    # Get sorted image IDs
    from metrics import sort_cameras_by_filename
    sorted_ids = sort_cameras_by_filename(reconstruction)
    
    # Extract camera poses
    trajectory = {}
    for image_id in sorted_ids:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        
        # Get camera position (translation vector)
        position = image.cam_from_world.translation
        
        # Get camera orientation as quaternion
        # Note: COLMAP uses a different convention, so we need to convert
        # Convert pycolmap.Rotation3d to numpy array
        quaternion = image.cam_from_world.rotation.todict()['quat']
        
        trajectory[frame_idx] = (position, quaternion)
    
    return trajectory

def get_frame_range(trajectory: Dict[FrameIndex, CameraPose], 
                    fmw) -> Tuple[int, int]:
    """
    Determine the full range of frame indices in the video.
    
    Parameters:
        trajectory: Dictionary of known camera poses
        fmw: FFmpegWrapper instance with video information
        
    Returns:
        Tuple of (min_frame_idx, max_frame_idx)
    """
    # Get the range from the known frames
    known_frames = list(trajectory.keys())
    min_known = min(known_frames)
    max_known = max(known_frames)
    
    # Get the total number of frames in the video
    total_frames = len(fmw.frames)
    
    # Return the full range (usually 1 to total_frames)
    return (1, total_frames)


def interpolate_positions(trajectory: Dict[FrameIndex, CameraPose], 
                          frame_indices: List[FrameIndex]) -> Dict[FrameIndex, Position]:
    """
    Interpolate camera positions for the given frame indices.
    
    Parameters:
        trajectory: Dictionary of known camera poses
        frame_indices: List of frame indices to interpolate
        
    Returns:
        Dictionary mapping frame indices to interpolated positions
    """
    # Extract known frame indices and positions
    known_indices = np.array(list(trajectory.keys()))
    known_positions = np.array([trajectory[idx][0] for idx in known_indices])
    
    # Create cubic spline interpolator for each coordinate (x, y, z)
    interpolators = [
        scipy.interpolate.CubicSpline(known_indices, known_positions[:, i])
        for i in range(3)
    ]
    
    # Interpolate positions for requested frames
    interpolated = {}
    for idx in frame_indices:
        if idx in trajectory:
            # Use known position if available
            interpolated[idx] = trajectory[idx][0]
        else:
            # Interpolate position
            pos = np.array([interp(idx) for interp in interpolators])
            interpolated[idx] = pos
    
    return interpolated

def interpolate_rotations(trajectory: Dict[FrameIndex, CameraPose], 
                          frame_indices: List[FrameIndex]) -> Dict[FrameIndex, Quaternion]:
    """
    Interpolate camera rotations for the given frame indices using quaternion SLERP.
    
    Parameters:
        trajectory: Dictionary of known camera poses
        frame_indices: List of frame indices to interpolate
        
    Returns:
        Dictionary mapping frame indices to interpolated quaternions
    """
    # Extract known frame indices and rotations
    known_indices = np.array(list(trajectory.keys()))
    known_rotations = [Rotation.from_quat(trajectory[idx][1]) for idx in known_indices]
    
    # Create rotation key frames for Slerp
    key_rots = Rotation.from_quat(np.array([r.as_quat() for r in known_rotations]))
    key_times = known_indices
    
    # Interpolate rotations for requested frames
    interpolated = {}
    for idx in frame_indices:
        if idx in trajectory:
            # Use known rotation if available
            interpolated[idx] = trajectory[idx][1]
        else:
            # Find the two nearest keyframes
            if idx < min(known_indices):
                # Before first keyframe, use first rotation
                interpolated[idx] = known_rotations[0].as_quat()
            elif idx > max(known_indices):
                # After last keyframe, use last rotation
                interpolated[idx] = known_rotations[-1].as_quat()
            else:
                # Find surrounding keyframes
                next_idx = np.searchsorted(known_indices, idx)
                prev_idx = next_idx - 1
                
                # Create a Slerp for just these two keyframes
                times = [known_indices[prev_idx], known_indices[next_idx]]
                rots = Rotation.from_quat([
                    known_rotations[prev_idx].as_quat(),
                    known_rotations[next_idx].as_quat()
                ])
                
                # Normalize the interpolation parameter
                t = (idx - times[0]) / (times[1] - times[0])
                
                # Interpolate
                slerp = Slerp(times, rots)
                interpolated[idx] = slerp([idx])[0].as_quat()
    
    return interpolated


def interpolate_camera_poses(reconstruction: pycolmap.Reconstruction, 
                            fmw,
                            confidence_falloff: float = 0.1) -> Dict[FrameIndex, Tuple[CameraPose, float]]:
    """
    Interpolate camera poses for all frames in the video sequence.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        fmw: FFmpegWrapper instance with video information
        confidence_falloff: Parameter controlling how quickly confidence decreases
                           with distance from known frames
        
    Returns:
        Dictionary mapping frame indices to (camera_pose, confidence)
    """
    # Extract known camera trajectory
    trajectory = extract_camera_trajectory(reconstruction)
    
    # Get full frame range
    min_frame, max_frame = get_frame_range(trajectory, fmw)
    all_frames = list(range(min_frame, max_frame + 1))
    
    # Interpolate positions and rotations
    positions = interpolate_positions(trajectory, all_frames)
    rotations = interpolate_rotations(trajectory, all_frames)
    
    # Combine into camera poses with confidence
    result = {}
    known_frames = set(trajectory.keys())
    
    for idx in all_frames:
        pose = (positions[idx], rotations[idx])
        
        if idx in known_frames:
            # Known frames have confidence 1.0
            confidence = 1.0
        else:
            # Find distance to nearest known frame
            distances = [abs(idx - k) for k in known_frames]
            nearest_distance = min(distances)
            
            # Compute confidence based on distance (exponential falloff)
            confidence = np.exp(-confidence_falloff * nearest_distance)
        
        result[idx] = (pose, confidence)
    
    return result

def validate_interpolation(reconstruction: pycolmap.Reconstruction, 
                          fmw,
                          validation_ratio: float = 0.2) -> Dict[str, float]:
    """
    Validate interpolation accuracy using leave-one-out cross-validation.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        fmw: FFmpegWrapper instance
        validation_ratio: Fraction of known frames to use for validation
        
    Returns:
        Dictionary with validation metrics
    """
    # Extract known camera trajectory
    full_trajectory = extract_camera_trajectory(reconstruction)
    known_frames = list(full_trajectory.keys())
    
    # Randomly select validation frames
    np.random.seed(42)  # For reproducibility
    n_validation = max(1, int(len(known_frames) * validation_ratio))
    validation_indices = np.random.choice(len(known_frames), n_validation, replace=False)
    validation_frames = [known_frames[i] for i in validation_indices]
    
    # Create a reduced trajectory without validation frames
    reduced_trajectory = {k: v for k, v in full_trajectory.items() if k not in validation_frames}
    
    # todo: clarify what is this metric exactly!
    
    # Create a reduced reconstruction
    # (This is a simplification - in practice, we'd need to create a new reconstruction object)
    from utils import _name_to_ind
    reduced_rec = copy.deepcopy(reconstruction)

    for img in reconstruction.images.values():
        idx = _name_to_ind(img.name)
        if idx in validation_frames:
            reduced_rec.deregister_image(img.image_id)
            del reduced_rec.images[img.image_id]
    
    # Interpolate poses using the reduced reconstruction
    interpolated = interpolate_camera_poses(reduced_rec, fmw)
    
    # Compute errors for validation frames
    position_errors = []
    rotation_errors = []
    
    for frame_idx in validation_frames:
        # Get ground truth pose
        true_position, true_rotation = full_trajectory[frame_idx]
        true_rotation_obj = Rotation.from_quat(true_rotation)
        
        # Get interpolated pose
        interp_pose, confidence = interpolated[frame_idx]
        interp_position, interp_rotation = interp_pose
        interp_rotation_obj = Rotation.from_quat(interp_rotation)
        
        # Compute position error (Euclidean distance)
        pos_error = np.linalg.norm(true_position - interp_position)
        position_errors.append(pos_error)
        
        # Compute rotation error (angle between rotations in degrees)
        rot_error = Rotation.from_quat(true_rotation).inv() * Rotation.from_quat(interp_rotation)
        angle_error = np.degrees(np.linalg.norm(rot_error.as_rotvec()))
        rotation_errors.append(angle_error)
    
    # Compute summary statistics
    metrics = {
        'mean_position_error': np.mean(position_errors),
        'max_position_error': np.max(position_errors),
        'mean_rotation_error_deg': np.mean(rotation_errors),
        'max_rotation_error_deg': np.max(rotation_errors),
    }
    
    return metrics
