# VMINer

## [Project page](https://costrice.github.io/vminer/) | [Video](https://www.youtube.com/watch?v=ry4frf6mIeA) | [Paper](https://costrice.github.io/pdfs/vminer.pdf) | [Data](https://www.dropbox.com/scl/fo/8cjuw3mhununvtaxac35k/AO-hoMtC4p8Q6mdR0hZur_A?rlkey=o1p4evpprjnyma60florhglnm&st=mkbeaw6r&dl=0)

Official implementation and project page of the CVPR'24 highlight paper
**VMINer: Versatile Multi-view Inverse Rendering with Near- and Far-field Light Sources**.

<p align="center">
  <img src="docs/static/images/teaser.png" width="80%" align="center">

## Setup

```bash
# Create a new conda environment
conda create -n vminer python=3.9
conda activate vminer

# Install and upgrade pip
conda install pip
python -m pip install --upgrade pip

# Install pytorch, tiny-cuda-nn, and other dependencies
# you can adjust according to your cuda version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

Tested on Windows 11 with one NVIDIA GeForce RTX 3090 (24GB) and CUDA 11.8.

## Preparing Data

We provide our synthetic and real dataset used in the paper,
which can be downloaded from the [Dropbox](https://www.dropbox.com/scl/fo/8cjuw3mhununvtaxac35k/AO-hoMtC4p8Q6mdR0hZur_A?rlkey=o1p4evpprjnyma60florhglnm&st=5dd6o41x&dl=0).
The scene name suffix `F*N*` denotes the number of far-field and near-field lights in the scene, respectively.

You can also generate your own dataset. Please refer to the [Data Format](#data-format) section and the script `scripts\convert_dataset_format.py` for more details.

### Data format

<details open>
<summary>click to expand/collapse</summary>

The data should be organized as follows:

```
<scene_name>
|-- train
    |-- light_metadata_train.json
    |-- train_000
        |-- rgba.png
        |-- metadata.json
    ...
|-- test
    |-- light_metadata_test.json
    |-- test_000
        |-- rgba.png
        |-- metadata.json
        |-- (optional) albedo.png
        |-- (optional) normal.png
        |-- (optional) rgb_diff.png
        |-- (optional) rgb_spec.png
    ...
```

The `light_metadata_*.json` files contain the overall lighting information, and is in the following format:

```json
{
    "far_lights": {
        "amount": <int>, // number of far-field lights
        "name": [
            <str>, ...  // names of the lighting
        ],
    },
    "near_lights": {
        "amount": <int>,  // number of near-field lights
        "pos_type": [  // types of near-field light positions
            "collocated",  // collocated with the camera
            "fixed",  // fixed position
            ...
        ]
    },
}
```

The `metadata.json` of each image contains camera pose and per-image lighting condition,
and is in the following format:

```json
{
    "cam_angle_x": <float>,  // horizontal field of view in radians
    "cam_transformation_matrix":
        // 4x4 camera extrinsic matrix from camera space  (opencv coordinate)
        // to world space
        ...,
    "imh": <int>,  // image height
    "imw": <int>,  // image width
    "far_light": <str>, // name of the far-field light that illuminates this image
    "near_light_status": [0, 1, 0, ...],  // on/off status of near-field lights
}
```

</details>

## Running the Code

Here we show how to run our code on one synthetic scene.
After downloading our synthetic data (for example, `hotdog/hotdog_F1N1`) to `/some/path`, you can run the following command to optimize the scene:

```bash
python main.py --config configs/train/hotdog/hotdog_F1N1.yaml --data_root /some/path
```

The visualization and the extracted textured mesh will be saved in `/log/workspace/*` by default.

If you want to render a reconstructed scene or extract meshes from a reconstructed scene checkpoint, an exemplar yaml is shown in `configs/hotdog_test.yaml`.

To run on your own data, please create a config file similar to `configs/train/*.yaml` then run the code with the config file.

You can also refer to `options.py` to see all the available options.

## Changelog

- 2024/06/30: Initial release

## Citation

If you find VMINer useful for your work, please consider citing:

```bibtex
@inproceedings{VMINer24,
    author       = {Fan Fei and
                    Jiajun Tang and
                    Ping Tan and
                    Boxin Shi},
    title        = {{VMINer}: Versatile Multi-view Inverse Rendering with Near- and Far-field Light Sources},
    booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                    {CVPR} 2024, Seattle, WA, USA, June 17-22, 2024},
    pages        = {11800-11809},
    publisher    = {{IEEE}},
    year         = {2023},
}
```

## Acknowledgement

When developing this project, we referred to the codebases of the following projects:

- [Instant-NSR (Pytorch)](https://github.com/zhaofuq/Instant-NSR)
- [TensoIR](https://github.com/Haian-Jin/TensoIR)
- [WildLight](https://github.com/za-cheng/WildLight)
- [mitsuba3](https://github.com/mitsuba-renderer/mitsuba3)
- [torch-spherical-harmonics](https://github.com/cheind/torch-spherical-harmonics/tree/main)

Thanks for their great work!

## Contacts

Please contact <feifan_eecs@pku.edu.cn> or open an issue for any questions or
suggestions.

Thanks! :smiley:
