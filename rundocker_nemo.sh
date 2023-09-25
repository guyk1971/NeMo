#!/bin/bash
# Define variables
IMAGE_NAME="nvcr.io/nvidia/nemo:23.04"

# Run the container
docker run --gpus all -it --rm -v $(pwd):/NeMo -v /home/guy/ssdb/datasets:/datasets --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd "$IMAGE_NAME"

