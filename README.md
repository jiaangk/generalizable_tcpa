This is the official repository of "Can Targeted Clean-label Poisoning Attacks Generalize?".

Our implementation is based on [Industrial Scale Data Poisoning via Gradient Matching](https://github.com/JonasGeiping/poisoning-gradient-matching).

# Environment

```bash
conda create --name gtp python=3.11 -y
conda activate gtp

pip install numpy==1.26.4
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# or 
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.39.3 scikit-learn==1.5.0

unzip expCUB.zip
unzip gims08.zip
```
The `expCUB.zip` and `gims08.zip` are our experiments' modified CUB-200-2011 and Multi-View Car datasets. They can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1Cf23kJZ43REMoFU4mlk_skov4qcZg3aB?usp=sharing).

# Evaluation

We present our paper experiments in the `experiments` folder. For example, to run the evaluation of the paper experiment in Table 1: 
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/table1_ours.sh
```

The result is saved as SQLite files. Some open-source software (e.g., DB Browser for SQLite, SQLiteStudio) can open them.
