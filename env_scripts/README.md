# Building and running Nemo container
the goal of this folder is to provide method to build and run nemo from container
There are 2 use cases:
1. local workstation - using `docker[_build]_run_lws.sh`
2. cluster - `docker[_build]_run_cl.sh`

the flow is similar, the differences is in the file

## Build Flow for local WS
### Build the docker and install NeMo
- ssh to workstation prompt
- `cd <NEMO ROOT>`
- `./env_scripts/docker_build_run_lws.sh` - this will build the image (if doesnt exit) and then run it  

from inside the container:
- `./reinstall.sh` - will install the NeMo from mapped source folder

At this point, you have a container with NeMo installed, ready for operation. 

### Commit to  the image
- Temporaly exit the container (ctrl+P+Q) and commit to the image:
```
gkoren@430f346ccb18:~/scratch/code/github/guyk1971/safari$ <press ctrl+p+q>
gkoren@ipp1-2161:~/safari$ docker commit <container_id> <image_name>
gkoren@ipp1-2161:~/safari$ docker push <image_name>  # optional
gkoren@ipp1-2161:~/safari$ docker attach <container_id>
```
where:
- `<container_id>` in this case is `430f346ccb18`  
- `image_name`= `<docker_repository>/<repo_name>:<updated_image_tag>`  


write down the `image_name` and update it in the `docker_run_lws.sh` and `docker_run_lws_multi.sh`.
in this case, I set the image name  `nemo_${MY_UNAME}:reinst`



### Running the docker image (reinst)
once the nemo has been installed within the image, you can run the `docker_run_lws.sh` script to create a singleton container (that has unique name and port mapped s.t. you can use tensorboard and jupyter)

if you want to run several containers in parallel, you need to remove any unique mapping/naming. for that, use the `docker_run_lws_multi.sh`