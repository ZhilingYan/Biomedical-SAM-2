# Biomedical SAM-2: Segment Anything in Biomedical Images and Videos

Biomedical SAM-2 (BioSAM-2) is an enhanced foundation model optimized for biomedical data based on SAM-2.

## Installation

Download code:
```python
git clone
cd Biomedical-SAM-2
```
Install the environment:
```python
conda env create -f environment.yml
conda activate BioSAM2
```
Then we need to download a model checkpoint. All the model checkpoints can be downloaded by running:
```python
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
or manually dowload Hiera-s from:
[sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)

## Model Training

Downloading [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) dataset from [here](https://drive.google.com/file/d/1q2_5K3PxWTYiwk_jAec37x-fMmawB9r4/view?usp=sharing)

Train and validate the dataset:
```python
python train.py -net sam2 -exp_name BUSI -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset BUSI -data_path BUSI_PATH -DiceCEloss True -nprompt click
```

## Cite
```bibtex
@article{yan2024biomedical,
  title={Biomedical sam 2: Segment anything in biomedical images and videos},
  author={Yan, Zhiling and Sun, Weixiang and Zhou, Rong and Yuan, Zhengqing and Zhang, Kai and Li, Yiwei and Liu, Tianming and Li, Quanzheng and Li, Xiang and He, Lifang and others},
  journal={arXiv preprint arXiv:2408.03286},
  year={2024}
}
```

## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [SAM-2](https://github.com/facebookresearch/segment-anything-2/tree/main) and [MedSAM-2](https://github.com/MedicineToken/Medical-SAM2/tree/main) for making their valuable code publicly available.
