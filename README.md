# FusionNet for Natural Language Inference

This is an example for applying FusionNet to natural language inference task.  
For more details on FusionNet, please refer to our paper:  
[FusionNet: Fusing via Fully-Aware Attention with Application to Machine Comprehension](https://arxiv.org/abs/1711.07341)  

Requirements
------------
+ Python (version 3.5.2)
+ PyTorch (0.2.0)
+ spaCy (1.x)
+ NumPy
+ JSON Lines
+ MessagePack

Since package update sometimes break backward compatibility, it is recommended to use Docker, which can be downloaded from [here](https://www.docker.com/community-edition#/download). To enable GPU, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) may also needs to be installed.  

After setting up Docker, simply perform `docker pull momohuang/fusionnet-docker` to pull the docker file. Note that this may take some time to download. Then we can run the docker image through  
`docker run -it momohuang/fusionnet-docker` (Only CPU)  
or  
`nvidia-docker run -it momohuang/fusionnet-docker` (GPU-enabled).  

Quick Start
-----------
`pip install -r requirements.txt`  
`bash download.sh`  
`python prepro.py`  
`python train.py`  
  
`train.py` supports an option `--full_att_type`, where  
`--full_att_type 0`: standard attention  
`--full_att_type 1`: fully-aware attention  
`--full_att_type 2`: fully-aware multi-level attention  
