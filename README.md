# Point Cloud Compression with Sibling Context and Surface Priors

> **Point Cloud Compression with Sibling Context and Surface Priors** <br>Zhili Chen, Zian Qian, Sukai Wang, Qifeng Chen <br>*European Conference on Computer Vision (ECCV) 2022*

[[Abs](https://arxiv.org/abs/2205.00760)][[Paper](https://arxiv.org/pdf/2205.00760.pdf)]

This repository is an official implementation of "[Point Cloud Compression with Sibling Context and Surface Priors](https://arxiv.org/abs/2205.00760)" (PCC-S). This is the initial version of PCC-S and this repo will be keep updated.

## Preparation

### Environment

```bash
conda create --name PCC-S python=3.8
conda activate PCC-S
sudo apt install build-essential python3-dev libopenblas-dev
conda install pytorch cudatoolkit -c pytorch
export CUDA_HOME=/usr/local/cuda-11.4; # or select the correct cuda version on your system.
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# To the root of the project folder
cd ..
bash requirements.sh
git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
```

### Data

1. Download the LiDAR point cloud data from [Kitti Odometry Dataset.](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

2. Set the absolute paths for variables `ROOT_dir`, `KITTI_BIN_dir`, and folder name for variable `PROCESSED_DATA_dir`for saving the preprocessed data in `config_ent.yml`.
   
   * `ROOT_dir`: Absolute path to this repo.
   
   * `KITTI_BIN_dir`: Absolute path to KITTI point cloud sequence.
   
   * `PROCESSED_DATA_dir`: Folder name for saving the preprocessed data.

3. Preprocess data with the following command. 

```bash
python util/preprocess_data.py --n_workers 16
```

## Compression

* Train the entropy model. Set the relative path for variable of `CKPT_DIR` to save checkpoint in `config_ent.yml`.

```bash
CUDA_VISIBLE_DEVICES='0,1' python PCCS.py
```

* Evaluate the compression performance. Set the folder name for variable `CKPT_DIR` to the trained checkpoint of the entropy model in `config_ent.yml` or set as `pretrained` to use the provided pretrained model.

```bash
CUDA_VISIBLE_DEVICES='0' python PCCS.py --is_validation
```

## Refinement

1. Set folder name for variable `pre_cache_feats_dir` in `config_ent.yml` for saving the cached predicted information from the entropy model. Then run the following command.
   
   ```bash
   CUDA_VISIBLE_DEVICES='0' python PCCS.py --to_cache_par_feats
   ```

2. Train the refinement model. Set the folder name for variable `CKPT_DIR` to save checkpoint in `config_refine.yml`.
   
   ```bash
   CUDA_VISIBLE_DEVICES='0,1' python refine.py
   ```

3. Evaluate the reconstruction performance. Set the folder name for variable `CKPT_DIR` to the trained checkpoint of the entropy model in `config_refine.yml` or set as `pretrained` to use the provided pretrained model.
   
   ```bash
   CUDA_VISIBLE_DEVICES='0' python refine.py --is_validation
   ```

## Encode and Decode

* Encode and decode. Set `compress_save_dir` in `config_ent.yml` for saving the encoded files and reconstructed point cloud in the form of `ply` .

```bash
CUDA_VISIBLE_DEVICES='0' python Encode_Decode.py
```

# Citation

If you find this project useful for your research, please consider citing:

```
@article{chen2022point,
  title={Point Cloud Compression with Sibling Context and Surface Priors},
  author={Chen, Zhili and Qian, Zian and Wang, Sukai and Chen, Qifeng},
  journal={arXiv preprint arXiv:2205.00760},
  year={2022}
}
```

# Contact

Feel free to contact me if there is any question. (Zhili Chen, leochenzhili@outlook.com )
