# Torch/nvidia base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime AS builder

COPY . /postera_ml_challenge_kinase
WORKDIR /postera_ml_challenge_kinase

# nvidia driver
    #Download Installer for Linux WSL-Ubuntu 2.0 x86_64
RUN sudo wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb && sudo dpkg -i cuda-keyring_1.0-1_all.deb && sudo apt-get update && sudo apt-get -y install cuda
# set up ssh server
RUN apt update && apt install openssh-server sudo -y && useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 matt && echo 'matt:postera' | chpasswd

# install requirements
RUN pip3 install --no-cache-dir pip install rdkit-pypi dgllife dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html -f https://data.pyg.org/whl/torch-1.13.0+cu116.html pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html torch-geometric

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime AS runner

COPY --from=builder . /
WORKDIR /postera_ml_challenge_kinase

RUN service ssh start

EXPOSE 22 8080
CMD ["/usr/sbin/sshd","-D"]
ENTRYPOINT ["python","main.py"]