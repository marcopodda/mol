# RUN WITH SOURCE, not BASH!!!

TORCH_VERSION=1.6.0
CUDA_VERSION=cu102
PYTHON_VERSION=3.7

# create venv
conda create --name chem python=${PYTHON_VERSION} -y && conda activate chem

# rm /home/podda/.miniconda3/envs/moldar/lib/libstdc++.so.6
# ln -s /home/podda/.miniconda3/envs/moldar/lib/libstdc++.so.6.0.26 /home/podda/.miniconda3/envs/moldar/lib/libstdc++.so.6

# install pytorch
conda install pytorch==${TORCH_VERSION} -c pytorch -y

# install pytorch-geometric
pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-geometric

# these dependencies seem to be needed
# conda install -c omgarcia libgcc-6 -y
# conda install gcc_linux-64 gxx_linux-64 -y

# install additional packages
conda install scikit-learn pandas joblib networkx pyyaml seaborn ipython jupyter -y

# additional pip packages
pip install molsets

# install rdkit
conda install rdkit -c rdkit -y
conda install molvs pytorch-lightning umap-learn -c conda-forge -y