docker run --gpus all -it \
    --name dab_detr \
    --shm-size=12g \
    -v ~/codespace/SAR-Specialized-Version-DAB-DETR/:/workspace \
    -v ~/datasets:/datasets \
    pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel \
    bash