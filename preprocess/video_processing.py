"""
Module: video_processing
Contains functionality for video metadata extraction and frame extraction.
"""

import subprocess
import os
import json
import shutil
from typing import Dict, List, Optional

class FFmpegWrapper:
    """
    Wrapper class for extracting video frames and metadata using FFmpeg.
    """
    def __init__(self, video_path: str, output_dir: str):
        """
        Initialize FFmpegWrapper by setting up directories, extracting metadata, and extracting frames.
        
        Parameters:
            video_path (str): Path to the input video.
            output_dir (str): Directory where extracted frames will be stored.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.tmp_path = os.path.join(os.path.dirname(video_path), "tmp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)
        self.fps, self.duration, self.width, self.height = self._get_video_metadata()
        self._extract_all_small_frames()
        self._get_frames_ids()
    
    def _get_video_metadata(self):
        """
        Retrieve FPS, duration, and resolution of the video using ffprobe.

        Returns:
            Tuple containing the frame rate, video duration, width and height.
        """
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration,width,height",
            "-of", "json", self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        stream = metadata['streams'][0]
        frame_rate = eval(stream['r_frame_rate'])
        duration = float(stream['duration'])
        width = int(stream['width'])
        height = int(stream['height'])
        return frame_rate, duration, width, height

    def _extract_all_small_frames(self):
        """
        Extract small frames (scaled to height 480) from the video if not already extracted.
        """
        if len(os.listdir(self.tmp_path)) == 0:
            extract_frames_cmd = f"ffmpeg -i {self.video_path} -pix_fmt rgb8 -q:v 4 -vf 'scale=-1:480' {self.tmp_path}/%08d.jpeg"
            exit_code = os.system(extract_frames_cmd)
            if exit_code != 0:
                print("error extracting frames")
                exit(exit_code)
        else:
            print("Frames already extracted")

    def _get_frames_ids(self):
        """
        Generate a sorted list of extracted frame filenames.
        """
        self.frames = sorted(os.listdir(self.tmp_path))

    def extract_specific_frames(self, frame_indices, full_res=False):
        """
        Extract specific frames from the video based on given frame indices.
        
        Parameters:
            frame_indices (list): List of frame indices to extract.
        """
        # Build select filter for ffmpeg command
        select_filter = "+".join([f"eq(n\\,{f})" for f in frame_indices])
        if full_res:
            scale_part = ""
        else:
            scale_part = ", scale=-1:960"

        cmd = f'ffmpeg -i {self.video_path} -vf "select={select_filter}{scale_part}" -vsync vfr -pix_fmt rgb8 -q:v 4 {self.output_dir}/%08d.jpeg'
        exit_code = os.system(cmd)
        if exit_code != 0:
            print("error extracting frames")
            exit(exit_code)

    def ind_to_frame_name(self, ind):
        """
        Convert a frame index to its corresponding filename.
        
        Parameters:
            ind (int): Frame index.
        
        Returns:
            str: Full path to the frame image.
        """
        name = "{:08d}.jpeg".format(ind)
        return os.path.join(self.tmp_path, name)

    def get_list_of_n_frames(self, n: int, start_frame: Optional[str] = None, end_frame: Optional[str] = None) -> List[str]:
        """
        Returns a list of n frame paths, equally distributed within the valid frame range.
        
        Parameters:
            n (int): Number of frames to select.
            start_frame (str, optional): Starting frame filename.
            end_frame (str, optional): Ending frame filename.
        
        Returns:
            List[str]: List of selected frame file paths.
        """
        if not self.frames or n <= 0:
            return []
        
        start_idx = self.frames.index(start_frame) if start_frame in self.frames else 0
        end_idx = self.frames.index(end_frame) if end_frame in self.frames else len(self.frames) - 1
        
        valid_frames = self.frames[start_idx:end_idx+1]
        total_frames = len(valid_frames)
        
        if n >= total_frames:
            print(f"Selected range has only {total_frames} frames. Returning all.")
            return [os.path.join(self.tmp_path, frame) for frame in valid_frames]
        
        step = total_frames / n
        indices = sorted(set(round(i * step) for i in range(n)))  # Ensure unique indices
        selected_frames = [valid_frames[i] for i in indices if i < total_frames]
        return [os.path.join(self.tmp_path, frame) for frame in selected_frames]
    
    def get_frames_between_pairs(self, peak_pairs: List[tuple], n: int) -> List[str]:
        """
        Returns a list of n frame paths between each specified pair of frames.
        
        Parameters:
            peak_pairs (List[tuple]): List of tuples containing pair of frame filenames.
            n (int): Number of frames to select between each pair.
        
        Returns:
            List[str]: List of selected frame file paths.
        """
        selected_frames = []
        for p1, p2 in peak_pairs:
            if p1 in self.frames and p2 in self.frames:
                selected_frames.extend(self.get_list_of_n_frames(n, start_frame=p1, end_frame=p2))
        return selected_frames


class ImageFolderWrapper:
    """Utility that mimics :class:`FFmpegWrapper` for pre-extracted image sequences."""

    _VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, images_path: str, output_dir: str):
        self.images_path = images_path
        self.output_dir = output_dir
        self.tmp_path = os.path.join(os.path.dirname(output_dir), "tmp")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)

        self.index_to_source: Dict[int, str] = {}
        self.index_to_tmp: Dict[int, str] = {}
        self.frames: List[str] = []

        self._prepare_frames()
        self.duration = len(self.frames)
        self.fps = None
        self.width = None
        self.height = None

    def _prepare_frames(self) -> None:
        existing = set(os.listdir(self.tmp_path))
        if existing:
            for name in sorted(existing):
                try:
                    idx = int(os.path.splitext(name)[0])
                except ValueError:
                    continue
                self.frames.append(name)
                self.index_to_tmp[idx] = name
            if self.frames:
                self.index_to_source = {
                    idx: os.path.join(self.images_path, self.index_to_tmp[idx])
                    for idx in self.index_to_tmp
                    if os.path.exists(os.path.join(self.images_path, self.index_to_tmp[idx]))
                }
                self.frames.sort()
                return

        shutil.rmtree(self.tmp_path)
        os.makedirs(self.tmp_path, exist_ok=True)

        images = [
            f for f in sorted(os.listdir(self.images_path))
            if f.lower().endswith(self._VALID_EXT)
        ]

        if not images:
            raise FileNotFoundError(f"No images found in {self.images_path}")

        for idx, name in enumerate(images):
            src = os.path.join(self.images_path, name)
            ext = os.path.splitext(name)[1].lower() or ".jpg"
            tmp_name = f"{idx:08d}{ext}"
            dst = os.path.join(self.tmp_path, tmp_name)
            shutil.copy2(src, dst)
            self.frames.append(tmp_name)
            self.index_to_source[idx] = src
            self.index_to_tmp[idx] = tmp_name

    def _frame_name(self, ind: int) -> Optional[str]:
        if ind in self.index_to_tmp:
            return self.index_to_tmp[ind]
        name = f"{ind:08d}"
        for candidate in self.frames:
            if candidate.startswith(name):
                self.index_to_tmp[ind] = candidate
                return candidate
        return None

    def get_list_of_n_frames(self, n: int, start_frame: Optional[str] = None, end_frame: Optional[str] = None) -> List[str]:
        if not self.frames or n <= 0:
            return []

        start_idx = self.frames.index(start_frame) if start_frame in self.frames else 0
        end_idx = self.frames.index(end_frame) if end_frame in self.frames else len(self.frames) - 1

        valid_frames = self.frames[start_idx:end_idx+1]
        total_frames = len(valid_frames)

        if n >= total_frames:
            return [os.path.join(self.tmp_path, frame) for frame in valid_frames]

        step = total_frames / n
        indices = sorted(set(round(i * step) for i in range(n)))
        selected_frames = [valid_frames[i] for i in indices if i < total_frames]
        return [os.path.join(self.tmp_path, frame) for frame in selected_frames]

    def extract_specific_frames(self, frame_indices: List[int], full_res: bool = False) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        for ind in frame_indices:
            src = self.index_to_source.get(ind)
            if src is None:
                frame_name = self._frame_name(ind)
                if frame_name is None:
                    continue
                candidate = os.path.join(self.images_path, frame_name)
                if os.path.exists(candidate):
                    src = candidate
            if src is None:
                continue
            ext = os.path.splitext(src)[1] or ".jpg"
            dst = os.path.join(self.output_dir, f"{ind:08d}{ext}")
            shutil.copy2(src, dst)

    def ind_to_frame_name(self, ind: int) -> str:
        frame = self._frame_name(ind)
        if frame is None:
            frame = f"{ind:08d}.jpg"
        return os.path.join(self.tmp_path, frame)

    def get_frames_between_pairs(self, peak_pairs: List[tuple], n: int) -> List[str]:
        selected_frames: List[str] = []
        for p1, p2 in peak_pairs:
            frame1 = self._frame_name(p1) if isinstance(p1, int) else p1
            frame2 = self._frame_name(p2) if isinstance(p2, int) else p2
            if frame1 in self.frames and frame2 in self.frames:
                selected_frames.extend(
                    self.get_list_of_n_frames(n, start_frame=frame1, end_frame=frame2)
                )
        return selected_frames
