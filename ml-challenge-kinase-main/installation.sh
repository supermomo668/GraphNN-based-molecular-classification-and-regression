# Raw setup (nvidia-cuda)
sudo apt install nvidia-driver-525
   # Meta package nvidia toolkit (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda


# python packages
  # nvcc
conda install -c "nvidia/label/cuda-11.6.2" cuda-nvcc
conda install pyg -c pyg
  # torch
pip install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  # dgl
pip install rdkit-pypi dgllife dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html 
  # pygeometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
