# VMINer

## [Project page](https://costrice.github.io/vminer/) | [Video](https://www.youtube.com/watch?v=ry4frf6mIeA) | [Paper](https://costrice.github.io/pdfs/vminer.pdf) | [Data](https://www.dropbox.com/scl/fo/8cjuw3mhununvtaxac35k/AO-hoMtC4p8Q6mdR0hZur_A?rlkey=o1p4evpprjnyma60florhglnm&st=mkbeaw6r&dl=0)

Official implementation and project page of the CVPR'24 highlight paper
**VMINer: Versatile Multi-view Inverse Rendering with Near- and Far-field Light Sources**.

<p align="center">
  <img src="docs/static/images/teaser.png" width="80%" align="center">

⚠️ 2024/08/06: Current code has some problem that will cause the training to fail after thousands of iterations due to
`RuntimeError: CUDA error: invalid configuration argument` when running on Linux (including the Docker image).
We are working on fixing this issue and will release a fixed version soon.

## Setup

### Using Conda
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

### Using Docker Image
We provide a built Docker image for running the code. You can pull the image from Docker Hub by running:
```bash
docker pull costrice/vminer:1.5
```

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

The light_metadata_*.jsonfiles contain the overall lighting information, and is in the following format:

```json
{
    "far_lights": {
        "amount": 1,
        "name": [
            "light1"
        ]
    },
    "near_lights": {
        "amount": 2,
        "pos_type": [  
            "collocated",  
            "fixed"
        ]
    }
}
```
Explanation:

- `far_lights`: This object contains information about the far-field lights.
    - `amount`: An integer that represents the number of far-field lights.
    - `name`: An array of strings where each string is the name of a far-field light.

- `near_lights`: This object contains information about the near-field lights.
    - `amount`: An integer that represents the number of near-field lights.
    - `pos_type`: An array of strings where each string represents the type of a near-field light position. The possible values are "collocated" (collocated with the camera) and "fixed" (fixed position).

The `metadata.json` of each image contains camera pose and per-image lighting condition,
and is in the following format:

```json
{
  "cam_angle_x": 0.6911112070083618,
  "cam_transformation_matrix": [],
  "imh": 800,
  "imw": 800,
  "far_light": "light1",
  "near_light_status": [1, 0]
}
```
Explanation:
- `cam_angle_x`: A float that represents the horizontal field of view in radians. 
- `cam_transformation_matrix`: A 4x4 matrix that represents the camera extrinsic matrix from camera space (in OpenCV coordinate) to world space. It is almost identical to the camera pose matrix in [NeRF format](https://github.com/bmild/nerf) except that the camera coordinate is in OpenCV coordinate rather than OpenGL coordinate, thus the 2nd and 3rd columns are flipped.
- `imh`: Image height in pixels.  
- `imw`: Image width in pixels.
- `far_light`: A string that represents the name of the far-field light that illuminates this image. The name should match one of the names specified in the `far_lights` section of the `light_metadata_*.json` file.
- `near_light_status`: An array of integers where each integer represents the on/off (1/0) status of a near-field light. The order of the statuses should match the order of the `pos_type` array in the near_lights section of the
  `light_metadata_*.json` file.

</details>

## Running the Code

Here we show how to run our code on one synthetic scene.
First, downloading our synthetic data (for example, `hotdog/hotdog_F1N1`) to `/path/to/data/root`.
Then run the command in subsequent sections in either Conda or Docker environment to optimize the scene.

We give some template configuration files in `configs/` folder.
For example, you can replace `/path/to/config/file.yaml`
with `configs/train/hotdog/hotdog_F1N1.yaml`.

### Using Conda
In conda environment, you can run the following command to optimize the scene:
```bash
python main.py \ 
  --config /path/to/config/file.yaml \
  --data_root /path/to/data/root \
  --scene_aabb $X1 $Y1 $Z1 $X2 $Y2 $Z2  # if you are using custom data
```

### Using Docker Image
Using Docker, you can run the image using the downloaded data with the following command:
```bash
docker run \
  --entrypoint python \
  --gpus device=0 \
  -v /path/to/data/root:/app/data/:ro \
  -v /path/to/config/file.yaml:/app/config.yaml:ro \
  -v ./log:/app/log/ \
  costrice/vminer:1.5 \
  main.py \
  --config config.yaml \
  --out_dir /app/log/workspace/ \
  --data_root /app/data/ \
  --scene_aabb $X1 $Y1 $Z1 $X2 $Y2 $Z2  # if you are using custom data
```

### Notes

The visualization and the extracted textured mesh will be saved
in `/log/workspace/$EXP_NAME-$TIMESTAMP` by default.

If you want to render a reconstructed scene or extract meshes from a
reconstructed scene checkpoint, an exemplar yaml is shown
in `configs/hotdog_test.yaml` (for running on Conda).

To run on your own data and configuration, you can either ([configargparse](https://github.com/bw2/ConfigArgParse) made this possible):
- create a config file similar
to `configs/train/*.yaml` then run the code with the config file using `--config /path/to/your/config.yaml`,
- or just pass the data root and other parameters directly to the command line using `--data_root`, `--scene_aabb`, etc.

Refer to `options.py` for all the available options.

## Changelog

- 2024/08/06: Release the docker image to run the code
- 2024/06/30: Release some scripts we used when writing the paper in `scripts/`. Currently, they are poorly documented and can not work out of the box. You can use them as references and see how the numbers in the paper are generated.
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
