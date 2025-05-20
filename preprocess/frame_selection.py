"""
Module: frame_selection
Contains advanced frame selection strategies for robust reconstruction.
"""

import numpy as np
import pycolmap
from typing import Dict, List, Tuple, Set, Optional
import os
import logging
from tqdm import tqdm

# Type aliases for clarity
Position = np.ndarray  # 3D position vector
Quaternion = np.ndarray  # Quaternion for rotation (x, y, z, w)
CameraPose = Tuple[Position, Quaternion]  # Combined position and orientation
FrameIndex = int  # Frame index in the video sequence
Point3DID = int  # ID of a 3D point in the reconstruction

from utils import _name_to_ind

def compute_frame_coverage_score(frame_idx: FrameIndex, 
                                visibility: Dict[Point3DID, float],
                                point_observations: Dict[Point3DID, Set[FrameIndex]],
                                total_points: int) -> float:
    """
    Compute a coverage score for a frame based on how many points it observes
    that are not well-observed by other frames.
    
    Parameters:
        frame_idx: Frame index
        visibility: Dictionary mapping point IDs to visibility scores for this frame
        point_observations: Dictionary mapping point IDs to sets of frames that observe them
        total_points: Total number of points in the reconstruction
        
    Returns:
        Coverage score (higher is better)
    """
    score = 0.0
    
    for point_id, vis_score in visibility.items():
        if vis_score > 0.5:  # Point is visible from this frame
            # Count how many other frames observe this point
            observation_count = len(point_observations.get(point_id, set()))
            
            # Points observed by fewer frames are more valuable
            if observation_count == 0:
                # New point, high value
                point_value = 1.0
            else:
                # Existing point, value decreases with observation count
                point_value = 1.0 / (observation_count + 1)
            
            # Add to score, weighted by visibility
            score += vis_score * point_value
    
    # Normalize by total points
    return score / total_points

def compute_frame_triangulation_score(frame_idx: FrameIndex,
                                     pose: CameraPose,
                                     all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                                     registered_frames: Set[FrameIndex],
                                     visibility: Dict[Point3DID, float],
                                     min_angle: float = 5.0,
                                     max_angle: float = 40.0) -> float:
    """
    Compute a triangulation score for a frame based on the viewing angles it forms
    with existing frames for visible points.
    
    Parameters:
        frame_idx: Frame index
        pose: Camera pose for this frame
        all_poses: Dictionary of all camera poses
        registered_frames: Set of already registered frame indices
        visibility: Dictionary mapping point IDs to visibility scores for this frame
        min_angle: Minimum good triangulation angle in degrees
        max_angle: Angle at which triangulation quality is maximized in degrees
        
    Returns:
        Triangulation score (higher is better)
    """
    if not registered_frames:
        return 0.0
    
    # Extract position of this frame
    position, _ = pose
    
    # Compute average triangulation angle with existing frames
    total_angle_score = 0.0
    count = 0
    
    for reg_frame in registered_frames:
        reg_pose, _ = all_poses[reg_frame]
        reg_position, _ = reg_pose
        
        # Vector between camera positions
        direction = reg_position - position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # Avoid division by zero
            continue
            
        # Normalize direction
        direction /= distance
        
        # For each visible point, compute triangulation angle
        for point_id, vis_score in visibility.items():
            if vis_score > 0.5:  # Point is visible
                # This is a simplification - in a real implementation, we would
                # use the actual 3D point position and compute proper viewing angles
                
                # For now, we'll use a heuristic based on camera distance
                # Assume triangulation angle is proportional to distance
                # This is just a placeholder - real implementation would be more sophisticated
                angle = min(max_angle, distance * 5.0)  # Simple heuristic
                
                # Score is highest when angle is at max_angle, lower for smaller or larger angles
                if angle < min_angle:
                    angle_score = angle / min_angle
                elif angle <= max_angle:
                    angle_score = 1.0
                else:
                    angle_score = max_angle / angle
                
                total_angle_score += angle_score * vis_score
                count += 1
    
    if count == 0:
        return 0.0
        
    return total_angle_score / count

def compute_frame_diversity_score(frame_idx: FrameIndex,
                                 pose: CameraPose,
                                 selected_frames: List[FrameIndex],
                                 all_poses: Dict[FrameIndex, Tuple[CameraPose, float]]) -> float:
    """
    Compute a diversity score for a frame based on how different it is from
    already selected frames.
    
    Parameters:
        frame_idx: Frame index
        pose: Camera pose for this frame
        selected_frames: List of already selected frame indices
        all_poses: Dictionary of all camera poses
        
    Returns:
        Diversity score (higher is better)
    """
    if not selected_frames:
        return 1.0  # First frame is maximally diverse
    
    # Extract position and orientation of this frame
    position, quaternion = pose
    
    # Compute minimum distance to any selected frame
    min_position_distance = float('inf')
    min_orientation_distance = float('inf')
    
    for sel_frame in selected_frames:
        sel_pose, _ = all_poses[sel_frame]
        sel_position, sel_quaternion = sel_pose
        
        # Position distance
        pos_distance = np.linalg.norm(position - sel_position)
        min_position_distance = min(min_position_distance, pos_distance)
        
        # Orientation distance (simplified)
        # In a real implementation, we would use proper quaternion distance
        ori_distance = np.linalg.norm(quaternion - sel_quaternion)
        min_orientation_distance = min(min_orientation_distance, ori_distance)
    
    # Normalize distances
    # These normalization factors would need to be tuned for your specific data
    norm_pos_distance = min(1.0, min_position_distance / 2.0)
    norm_ori_distance = min(1.0, min_orientation_distance / 0.5)
    
    # Combine position and orientation diversity
    return 0.7 * norm_pos_distance + 0.3 * norm_ori_distance

def compute_frame_confidence_score(frame_idx: FrameIndex,
                                  all_poses: Dict[FrameIndex, Tuple[CameraPose, float]]) -> float:
    """
    Get the confidence score for a frame's interpolated pose.
    
    Parameters:
        frame_idx: Frame index
        all_poses: Dictionary of all camera poses with confidence scores
        
    Returns:
        Confidence score (higher is better)
    """
    _, confidence = all_poses[frame_idx]
    return confidence

def select_best_frames(
    reconstruction: pycolmap.Reconstruction,
    all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
    all_visibility: Dict[FrameIndex, Dict[Point3DID, float]],
    n_frames: int = 50,
    coverage_weight: float = 0.4,
    triangulation_weight: float = 0.3,
    diversity_weight: float = 0.2,
    confidence_weight: float = 0.1,
    prune_percentile: float = 10.0,
) -> List[FrameIndex]:
    # 1) build registered frames & point→obs map (unchanged)
    registered: Set[int] = set()
    for img in reconstruction.images.values():
        registered.add(_name_to_ind(img.name))

    point_observations: Dict[Point3DID, Set[FrameIndex]] = {}
    for img in reconstruction.images.values():
        fidx = _name_to_ind(img.name)
        for p2d in img.points2D:
            if p2d.has_point3D():
                pid = p2d.point3D_id
                point_observations.setdefault(pid, set()).add(fidx)

    total_points = max(1, len(reconstruction.points3D))

    # 2) Precompute static_score for each candidate frame
    static_scores = {}
    # also stash positions & quats into arrays for fast diversity
    cand_positions = {}
    cand_quats = {}
    for fidx, (pose, conf) in all_poses.items():
        if fidx in registered or fidx not in all_visibility:
            continue
        vis_dict = all_visibility[fidx]
        if not vis_dict:
            continue

        # --- vectorized coverage score ---
        pts = np.array(list(vis_dict.keys()), dtype=int)
        vis = np.array([vis_dict[p] for p in pts], dtype=float)
        # only keep those > .5
        mask = vis > 0.5
        if mask.any():
            obs_counts = np.array([len(point_observations.get(p, ())) for p in pts], dtype=float)
            # new points get 1/(0+1)=1
            values = 1.0 / (obs_counts + 1.0)
            cov_score = (vis[mask] * values[mask]).sum() / total_points
        else:
            cov_score = 0.0

        # --- static triangulation w.r.t. already-registered frames ---
        # collect all registered positions once into array
        if registered:
            reg_poses = [all_poses[r][0][0] for r in registered]
            reg_pos_arr = np.stack(reg_poses, axis=0)  # (R,3)
            pos, _ = pose
            # distances to each registered cam
            dists = np.linalg.norm(reg_pos_arr - pos[None,:], axis=1)
            # simple heuristic: angle_i = clip(dist_i*5, min_angle, max_angle)
            angles = np.clip(dists * 5.0, 5.0, 40.0)
            # score per reg-camera ~1 when in [min,max], else linear drop
            angle_scores = np.where(
                angles < 5.0, angles/5.0,
                np.where(angles <= 40.0, 1.0, 40.0/angles)
            )
            # weight by avg visibility
            avg_vis = vis.mean() if vis.size>0 else 0.0
            tri_score = (angle_scores.mean() * avg_vis)
        else:
            tri_score = 0.0

        # --- confidence score ---
        conf_score = conf

        static_scores[fidx] = (
            coverage_weight * cov_score
            + triangulation_weight * tri_score
            + confidence_weight * conf_score
        )

        # stash pose for diversity:
        cand_positions[fidx] = pose[0]
        cand_quats[fidx]     = pose[1]

    if not static_scores:
        return []

    # 3) Early prune: drop bottom X% by static_score
    scores_arr = np.array(list(static_scores.values()))
    thresh = np.percentile(scores_arr, prune_percentile)
    candidates = [f for f, s in static_scores.items() if s >= thresh]

    # initialize diversity arrays
    M = len(candidates)
    min_pos_dist = {f: np.inf for f in candidates}
    min_ori_dist = {f: np.inf for f in candidates}

    selected = []

    # 4) Greedy selection with only diversity updates per-iteration
    for _ in tqdm(range(min(n_frames, len(candidates))), desc="Selecting frames"):
        # compute total_score = static_score + diversity_weight * diversity_score
        total_scores = {}
        for f in candidates:
            if f in selected:
                continue
            # normalize
            npd = min(1.0, min_pos_dist[f]/2.0)
            nod = min(1.0, min_ori_dist[f]/0.5)
            div_score = 0.7*npd + 0.3*nod
            total_scores[f] = static_scores[f] + diversity_weight * div_score

        # pick best
        best = max(total_scores, key=total_scores.get)
        selected.append(best)

        # update diversity distances in one pass
        bp = cand_positions[best]
        bq = cand_quats[best]
        for f in candidates:
            if f in selected:
                continue
            # position
            dpos = np.linalg.norm(cand_positions[f] - bp)
            if dpos < min_pos_dist[f]:
                min_pos_dist[f] = dpos
            # orientation (L2 on quaternion)
            dori = np.linalg.norm(cand_quats[f] - bq)
            if dori < min_ori_dist[f]:
                min_ori_dist[f] = dori

    return selected

def select_random_frames(all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                        registered_frames: Set[FrameIndex],
                        selected_frames: List[FrameIndex],
                        n_frames: int = 10) -> List[FrameIndex]:
    """
    Select random frames that are not already registered or selected.
    
    Parameters:
        all_poses: Dictionary of all camera poses
        registered_frames: Set of already registered frame indices
        selected_frames: List of already selected frame indices
        n_frames: Number of random frames to select
        
    Returns:
        List of randomly selected frame indices
    """
    # Get all available frames
    available_frames = [idx for idx in all_poses.keys() 
                       if idx not in registered_frames and idx not in selected_frames]
    
    if not available_frames:
        return []
    
    # Select random frames
    np.random.seed(42)  # For reproducibility
    n_select = min(n_frames, len(available_frames))
    random_indices = np.random.choice(len(available_frames), n_select, replace=False)
    
    return [available_frames[i] for i in random_indices]

def adaptive_frame_selection(reconstruction: pycolmap.Reconstruction,
                            all_poses: Dict[FrameIndex, Tuple[CameraPose, float]],
                            all_visibility: Dict[FrameIndex, Dict[Point3DID, float]],
                            coverage_weight,
                            triangulation_weight,
                            diversity_weight,
                            confidence_weight,
                            n_frames: int = 50,
                            random_ratio: float = 0.2,) -> List[FrameIndex]:
    """
    Select frames for reconstruction using an adaptive strategy that combines
    best frame selection with random sampling.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        all_poses: Dictionary mapping frame indices to (camera_pose, confidence)
        all_visibility: Dictionary mapping frame indices to dictionaries of point visibility scores
        n_frames: Total number of frames to select
        random_ratio: Ratio of frames to select randomly
        
    Returns:
        List of selected frame indices
    """
    
    # Get registered frames
    registered_frames = set()
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        registered_frames.add(frame_idx)
    
    # Determine number of frames to select based on strategy
    n_best_frames = int(n_frames * (1 - random_ratio))
    n_random_frames = n_frames - n_best_frames
    
    # Select best frames
    best_frames = select_best_frames(
        reconstruction,
        all_poses,
        all_visibility,
        n_frames=n_best_frames,
        coverage_weight=coverage_weight,
        triangulation_weight=triangulation_weight,
        diversity_weight=diversity_weight,
        confidence_weight=confidence_weight
    )
    
    # Select random frames
    random_frames = select_random_frames(
        all_poses, registered_frames, best_frames, n_frames=n_random_frames)
    
    # Combine and return
    selected_frames = best_frames + random_frames
    return selected_frames

def analyze_frame_contribution(reconstruction: pycolmap.Reconstruction,
                              frame_indices: List[FrameIndex]) -> Dict[FrameIndex, Dict[str, float]]:
    """
    Analyze the contribution of each frame to the reconstruction quality.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        frame_indices: List of frame indices to analyze
        
    Returns:
        Dictionary mapping frame indices to dictionaries of contribution metrics
    """
    from metrics import compute_overlaps_in_rec
    
    # Get mapping from frame indices to image IDs
    frame_to_image = {}
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        frame_to_image[frame_idx] = image_id
    
    # Compute baseline metrics
    baseline_overlaps, _ = compute_overlaps_in_rec(reconstruction)
    baseline_min = np.min(baseline_overlaps) if baseline_overlaps else 0
    baseline_avg = np.mean(baseline_overlaps) if baseline_overlaps else 0
    baseline_points = len(reconstruction.points3D)
    
    # Analyze contribution of each frame
    contributions = {}
    for frame_idx in frame_indices:
        if frame_idx not in frame_to_image:
            # Frame not in reconstruction
            contributions[frame_idx] = {
                'min_overlap_change': 0.0,
                'avg_overlap_change': 0.0,
                'points_change': 0.0,
                'total_score': 0.0
            }
            continue
        
        image_id = frame_to_image[frame_idx]
        
        # Create a copy of the reconstruction without this frame
        rec_copy = pycolmap.Reconstruction(reconstruction)
        rec_copy.deregister_image(image_id)
        
        # Compute metrics without this frame
        new_overlaps, _ = compute_overlaps_in_rec(rec_copy)
        new_min = np.min(new_overlaps) if new_overlaps else 0
        new_avg = np.mean(new_overlaps) if new_overlaps else 0
        new_points = len(rec_copy.points3D)
        
        # Compute contribution metrics
        min_overlap_change = baseline_min - new_min
        avg_overlap_change = baseline_avg - new_avg
        points_change = baseline_points - new_points
        
        # Normalize metrics
        norm_min_overlap = min_overlap_change / (baseline_min + 1e-6)
        norm_avg_overlap = avg_overlap_change / (baseline_avg + 1e-6)
        norm_points = points_change / (baseline_points + 1e-6)
        
        # Compute total score
        total_score = (
            0.3 * norm_min_overlap +
            0.3 * norm_avg_overlap +
            0.4 * norm_points
        )
        
        contributions[frame_idx] = {
            'min_overlap_change': min_overlap_change,
            'avg_overlap_change': avg_overlap_change,
            'points_change': points_change,
            'total_score': max(0.0, total_score)
        }
    
    return contributions

def prune_frames(reconstruction: pycolmap.Reconstruction,
                threshold: float = 0.05) -> Tuple[pycolmap.Reconstruction, List[FrameIndex]]:
    """
    Remove frames that don't contribute significantly to the reconstruction.
    
    Parameters:
        reconstruction: COLMAP reconstruction object
        threshold: Minimum contribution threshold
        
    Returns:
        Tuple of (pruned reconstruction, list of removed frame indices)
    """
    
    # Get frame indices
    frame_indices = []
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        frame_indices.append(frame_idx)
    
    # Analyze frame contributions
    contributions = analyze_frame_contribution(reconstruction, frame_indices)
    
    # Identify frames to remove
    frames_to_remove = [idx for idx, contrib in contributions.items() 
                       if contrib['total_score'] < threshold]
    
    print(f"Removing {len(frames_to_remove)} low-contribution frames out of {len(frame_indices)}")
    
    # Create a copy of the reconstruction
    pruned_rec = pycolmap.Reconstruction(reconstruction)
    
    # Remove low-contribution frames
    frame_to_image = {}
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        frame_idx = _name_to_ind(image.name)
        frame_to_image[frame_idx] = image_id
    
    for frame_idx in frames_to_remove:
        if frame_idx in frame_to_image:
            image_id = frame_to_image[frame_idx]
            pruned_rec.deregister_image(image_id)
    
    return pruned_rec, frames_to_remove