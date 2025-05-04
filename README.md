# How to get running on lapotp or Desktop - if no conda then skip lines 1-6

Code from https://www.youtube.com/watch?v=Vonyoz6Yt9c with github link https://github.com/uygarkurt/ViT-PyTorch

1. ```mkdir -p ~/miniconda3```
2. download the correct version of miniconda for the architecture
    - ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh``` #x86_64
    - ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh``` #ARM64 (you may also need to ```export PATH="$HOME/miniconda3/bin:$PATH"```)
3. ```bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3```
4. ```rm ~/miniconda3/miniconda.sh```
5. ```conda init bash```
6. ```source ~/.bashrc```
7. ```conda create -n handwriting_venv```
8. ```conda activate handwriting_venv```
8. ```conda install python```
9. ```pip install numpy scikit-learn tqdm matplotlib pandas```
10. dependent upon the architecture
    - ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126``` #x86_64
    - ```pip install torch torchvision torchaudio``` #ARM64
11. ```conda install nvidia/label/cuda-12.0.0::cuda-toolkit```
12. clone the repository
    - ```git clone https://github.com/BobtheElf/handwriting_transformer_for_performance_comparison.git``` #cd into the directory you would like to run from
    - Generate SSH keys
        1. login to github
        2. ```ssh-keygen -t ed25519 -C "your_email@example.com"``` #make ssh keypair on computer you would like the repository to go into
        3. ```git clone git@github.com:BobtheElf/handwriting_transformer_for_performance_comparison.git```
13. ```python ./test.py```
14. (upon successful test.py) ```python ./handwriting.py```

# How to get running on NVIDIA edge device
1. ```wget -qO- https://astral.sh/uv/install.sh```
2. ```uv venv .venv --system-site-packages --python=3.10```
3. ```source .venv/bin/activate```
4. ```uv pip install jetsontools```
5. ```uv pip install numpy==1.25.0 scikit-learn tqdm matplotlib pandas```
6. on edge device, you may be maxing out memory, so be sure to clean up if you run into this issue
    - commands:
        - ```df -h```
        - ```conda clean --all --yes```
        - ```rm -rf ~/miniconda3/pkgs/*.conda```
        - ```rm -rf ~/miniconda3/pkgs/*.tar.bz2```
        - ```rm -rf ~/.cache/pip```
        - ```rm -rf ~/.cache/torch```
        - ```rm -rf ~/.cache/huggingface```
12. clone the repository
    - ```git clone https://github.com/BobtheElf/handwriting_transformer_for_performance_comparison.git``` #cd into the directory you would like to run from
    - Generate SSH keys
        1. login to github
        2. ```ssh-keygen -t ed25519 -C "your_email@example.com"``` #make ssh keypair on computer you would like the repository to go into
        3. ```git clone git@github.com:BobtheElf/handwriting_transformer_for_performance_comparison.git```
13. ```python ./test.py```
14. (upon successful test.py) ```python ./handwriting.py```