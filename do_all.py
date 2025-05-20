import os
import shutil
from argparse import ArgumentParser
import time

CURR_PATH = os.path.dirname(__file__)

def get_video_length(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(frame_count / video_fps)

    return duration, frame_count, video_fps

def do_one(source_p, n_frames, clean=False, minimal=False, full=False):

    start_time = time.time()

    files_n = os.listdir(source_p)
    video_n = None
    for f in files_n:
        if f.split(".")[-1] in ["mp4", "MP4"]:
            video_n = f
            break

    if video_n is None and (not ("input" in files_n)):
        exit(1)


    images_p = os.path.join(source_p, 'images')
    sparse_p = os.path.join(source_p, 'sparse')
    model_p = os.path.join(source_p, 'model')
    depths_p = os.path.join(source_p, 'd_images')


    if (not (os.path.isdir(images_p) and os.path.isdir(sparse_p)) or clean):
        # extract frames, and perform SFM from video
        sfm_command = f"python preprocess/main_video_process.py -s {source_p} -n {n_frames} --robust"
        
        if clean:
            sfm_command += " -c"
        if minimal:
            sfm_command += " -m"
        if full:
            sfm_command += " -f"
            
        print(sfm_command)
        exit_code = os.system(sfm_command)
        if exit_code != 0:
            print("error while performing sfm")
            exit(exit_code)

    sfm_time = time.time()

    if not os.path.isdir(depths_p):
        # estimate depth maps
        depth_command = f"cd {CURR_PATH}/submodules/DepthAnythingV2_docker/ && python run.py --encoder vitl --pred-only --grayscale --img-path {images_p} --outdir  {depths_p}"
        exit_code = os.system(depth_command)
        if exit_code != 0:
            print("error while performing depth estimation")
            exit(exit_code)

    # estimating depth scale
    scale_cmd = f"conda run -n gaussian_splatting python {CURR_PATH}/submodules/gaussian-splatting/utils/make_depth_scale.py --base_dir {source_p} --depths_dir {depths_p}"
    exit_code = os.system(scale_cmd)
    if exit_code != 0:
        print("error while performing depth scale estimation")
        exit(exit_code)

    depth_time = time.time()

    train_cmd = f"conda run -n gaussian_splatting python {CURR_PATH}/submodules/gaussian-splatting/train.py " + \
        f"-s {source_p} -m {model_p} -d {depths_p} " + \
        "--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp " + \
        "--data_device cpu --optimizer_type sparse_adam   --antialiasing"
    
    if not full:
        train_cmd += " --iterations 7000"

    print(f"Started trainig with cmd:\n{train_cmd}")
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("error while training")
        exit(exit_code)

    end_time = time.time()

    print(f"Total time: {end_time - start_time};\nsfm time: {sfm_time - start_time}\ndepth time: {depth_time - sfm_time}\ngs time: {end_time - depth_time}")


def main(args):

    source_p = args.source_path
    n_frames = args.max_number_of_frames
    clean = args.clean
    minimal = args.minimal
    full = args.full
    if not args.all:
        do_one(source_p, n_frames, clean=clean, minimal=minimal, full=full)
    else:
        dirs = os.listdir(source_p)
        for d in dirs:
            tmp = os.path.join(source_p, d)
            if not os.path.isdir(tmp):
                continue
            do_one(tmp, n_frames, clean=clean, minimal=minimal)



if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--max_number_of_frames", "-n", default=400, type=int)
    parser.add_argument("--clean", "-c", action='store_true')
    parser.add_argument("--minimal", "-m", action='store_true', help="Use minimal frame selection after final reconstruction")
    parser.add_argument("--full", "-f", action='store_true', help="Use all frame selection after final reconstruction")
    parser.add_argument("--all", "-a", action='store_true')
    args = parser.parse_args()

    main(args)
