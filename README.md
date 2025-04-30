# How to get running - if no conda then skip lines 1-6

Code from https://www.youtube.com/watch?v=Vonyoz6Yt9c with github link https://github.com/uygarkurt/ViT-PyTorch

1. mkdir -p ~/miniconda3
2. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
3. bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
4. rm ~/miniconda3/miniconda.sh
5. conda init bash
6. source ~/.bashrc
7. conda create -n handwriting_venv
8. conda activate handwriting_venv
8. conda install python
9. pip install numpy scikit-learn tqdm matplotlib pandas
10. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
11. conda install nvidia/label/cuda-12.0.0::cuda-toolkit
12. git clone https://github.com/BobtheElf/handwriting_transformer_for_performance_comparison.git #cd into the directory you would like to run from
13. python ./test.py
14. (upon successful test.py) python ./handwriting.py