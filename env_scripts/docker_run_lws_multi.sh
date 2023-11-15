#!/bin/bash
# Define variables

#!/bin/bash
# DIR is the directory where the script is saved (should be <project_root/scripts)
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $DIR

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)
IMAGE=nemo_${MY_UNAME}:dev
# mount the scratch folders : assuming you have a relative soft link to scratch created by  'ln -s ../scratch.gkoren_gpu scratch'
EXTRA_MOUNTS=""
if [ -d "/home/${MY_UNAME}/scratch" ]; then
    EXTRA_MOUNTS+=" --mount type=bind,source=/home/${MY_UNAME}/scratch,target=/home/${MY_UNAME}/scratch"
fi
# if you have another scratch : 
if [ -d "/home/${MY_UNAME}/scratch_1" ]; then
    EXTRA_MOUNTS+=" --mount type=bind,source=/home/${MY_UNAME}/scratch_1,target=/home/${MY_UNAME}/scratch_1"
fi

docker run \
    --gpus \"device=all\" \
    --privileged \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --mount type=bind,source=${DIR}/..,target=${DIR}/.. \
    --shm-size=8g \
    ${IMAGE}

#     ${EXTRA_MOUNTS} \

cd -
