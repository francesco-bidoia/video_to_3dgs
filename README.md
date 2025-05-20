# video_to_3dgs
This  repo contains an end to end pipeline to generate 3dgs from videos 
# Install
We use Depth Anything v2 to generate depth estimates. 
Download Depth-Anything-V2-Large from here: https://github.com/DepthAnything/Depth-Anything-V2 and place it here: submodules/DepthAnythingV2_docker/checkpoints/depth_anything_v2_vitl.pth

We provide a dockerfile and a start script.
Simply run ./start.sh to enter the container (it will build it automatically)


# How to use
We assume there is a fodler called dataset_gs at the same level as this repo.

Create a folder under dataset_gs and add a .mp4 video file to it:
dataset_gs/example1/anyVideoName.mp4

To run the pipeline use python do_all.py -s /v2gs/datasets_gs/example1 -n n_of_images