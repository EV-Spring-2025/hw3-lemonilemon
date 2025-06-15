[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SdXSjEmH)

# EV-HW3: PhysGaussian

## Homework Specification

This homework is based on the recent CVPR 2024 paper [PhysGaussian](https://github.com/XPandora/PhysGaussian/tree/main), which introduces a novel framework that integrates physical constraints into 3D Gaussian representations for modeling generative dynamics.

You are **not required** to implement training from scratch. Instead, your task is to set up the environment as specified in the official repository and run the simulation scripts to observe and analyze the results.

### Getting the Code from the Official PhysGaussian GitHub Repository

Download the official codebase using the following command:

```
git clone https://github.com/XPandora/PhysGaussian.git
```

### Environment Setup

Navigate to the "PhysGaussian" directory and follow the instructions under the "Python Environment" section in the official README to set up the environment.

### Running the Simulation

Follow the "Quick Start" section and execute the simulation scripts as instructed. Make sure to verify your outputs and understand the role of physics constraints in the generated dynamics.

### Homework Instructions

Please complete Part 1–2 as described in the [Google Slides](https://docs.google.com/presentation/d/13JcQC12pI8Wb9ZuaVV400HVZr9eUeZvf7gB7Le8FRV4/edit?usp=sharing).

## My Answer (Results)

I used the offical PhysGaussian repository on CSIE workstation Meow1 to run the simulation. However, I encountered some issues with the environment setup, the way I solved the issues is described in the following sections. The results of the simulation are also included below.

### Environment Setup (on Meow1)
As the CUDA version on Meow1 is 12.8 which is not the desired version for the PhysGaussian repository, I modified the codebase to make it compatible with the newer version (e.g. include corresponding C libraries, increase memory limits, etc.) . I forked the official repository and made the necessary changes to the codebase. In that case, after cloning the repository, one can directly use the update the submodule from the forked repository:

```bash

```

After that, I use `venv` to create a virtual environment using the `requirement.txt` I provided and the version of Python is 3.9.23.

Finally, install the PhysGaussian repository as a package in the virtual environment as the official repository suggests:

```bash
cd PhysGaussian
pip install -e .
```

And hopefully, you can run the simulation scripts without any issues on the Meow1 workstation.

### Part 1: Baseline Simulation
In this part, I simulated two different materials with default parameters as a baseline, as required by the assignment. The result is shown below:

- `sand`:
    - Baseline Parameter Settings:
        | Parameter | Value |
        |:----------|:---------------------|
        | `material` | `sand` |
        | `n_grid` | 100 |
        | `substep_dt` | 1e-4 |
        | `grid_v_damping_scale` | 0.9999 |
        | `softening` | 0.1 |

        **Note:** For the details of other parameters, please refer to the file I provided in `config/baseline/custom_sand_sim.json`. The `n_grid` is larger than other materials, which is because the sand material requires a higher resolution to simulate the details of the sand particles.

    - Simulation Video: [Link to the video](https://youtube.com/shorts/2JhbbjYjp6c?feature=share)

        ![GIF](https://github.com/user-attachments/assets/8fcc079e-9af9-4634-b71a-0a6d60168805)
    - Brief Description:
        `sand` is a granular medium. Instead of a single solid object, it behaves as a collection of individual particles. Upon impact, the sand does not bounce as a whole but instead flows and disperses, with particles scattering and settling into a pile. This demonstrates the characteristic behavior of granular materials.
- `metal`:
    - Parameters:
        | Parameter | Value |
        |:----------|:---------------------|
        | `material` | `metal` |
        | `n_grid` | 25 |
        | `substep_dt` | 1e-4 |
        | `grid_v_damping_scale` | 0.9999 |
        | `softening` | 0.1 |

        **Note:** For the details of other parameters, please refer to the file I provided in `config/baseline/custom_metal_sim.json`.
    - Simulation Video: [Link to the video](https://youtube.com/shorts/hDyMcZAmI74?feature=share)

        ![GIF](https://github.com/user-attachments/assets/2a4f90f9-7135-4c97-92b2-ee5788c1a7a2)
    - Brief Description:
        `metal` behaves as an elastoplastic solid. Under small impacts, it is elastic, meaning it resists bending and springs back to its original shape. However, if subjected to a very strong force, it will exceed its elastic limit and undergo plastic deformation, causing it to bend permanently and not return to its original form. In this `ficus` simulation, as I don't provide a `yield_stress` parameter, it defaults to be 0, which means the metal will start to deform plastically under any impact, resulting in a permanent bend.
- `plasticine`:
    - Parameters:
        | Parameter | Value |
        |:----------|:---------------------|
        | `material` | `plasticine` |
        | `n_grid` | 25 |
        | `substep_dt` | 1e-4 |
        | `grid_v_damping_scale` | 0.9999 |
        | `softening` | 0.1 |
        | `E` | 1e4 |

        **Note:** For the details of other parameters, please refer to the file I provided in `config/baseline/custom_plasticine_sim.json`.
    - Simulation Video: [Link to the video](https://youtube.com/shorts/iPZ-AKoJf8w?feature=share)

        ![GIF](https://github.com/user-attachments/assets/60b86fca-22c5-4f30-9b28-3e3502a00c87)
    - Brief Description:

The default values of the parameters are mostly from `ficus_config.json` in the official repository, except for the parameters that are listed above. And all the simulation videos are run with the original `gs_simlation.py` with the `ficus_whitebg-trained` model:

```bash
python gs_simulation.py --model_path model/ficus_whitebg-trained/ --output_path <output_path> --config ../config/<config_path> --render_img --compile_video
```

---

### Part 2: Exploring MPM Parameter Effects

This section details the results of an ablation study on the key physical parameters in the simulation. The study is organized by parameter to analyze its effect across different materials.

PSNR curves are used to evaluate the difference between the baseline and adjusted parameters. It is computed with my own implementation of PSNR, you can run it as follows:

```bash
```

#### 1. Adjusting `n_grid`

This study explores the effect of the MPM grid resolution (`n_grid`).

* **Material:** `sand` (decreased from 100 to 50)
    * **PSNR:** 29.05

        ![PSNR vs Frame](imgs/n_grid/sand.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/pRCkLeAwtV4?feature=share)
* **Material:** `metal` (increased from 25 to 50)
    * **PSNR:** 17.01

        ![PSNR vs Frame](imgs/n_grid/metal.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/hDyMcZAmI74?feature=share)
* **Material:** `plasticine` (increased from 25 to 50)
    * **PSNR:** 22.08

        ![PSNR vs Frame](imgs/n_grid/plasticine.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/UkUaPRoWHSk?feature=share)

##### Visual Comparison for `n_grid`

| Material    | Baseline (`n_grid`: default)                 | Adjusted (`n_grid`: 50)                      |
| :---------- | :-------------------------------------------: | :-------------------------------------------: |
| **Sand** | ![Baseline Sand GIF](https://github.com/user-attachments/assets/8fcc079e-9af9-4634-b71a-0a6d60168805) | ![Adjusted n_grid Sand GIF](https://github.com/user-attachments/assets/1e9e694d-016b-4da6-aa0f-4c259578b109) |
| **Metal** | ![Baseline Metal GIF](https://github.com/user-attachments/assets/2a4f90f9-7135-4c97-92b2-ee5788c1a7a2) | ![Adjusted n_grid Metal GIF](https://github.com/user-attachments/assets/79baaa0b-6d0e-4746-83f4-a9cb40ac19a1) |
| **Plasticine** | ![Baseline Plasticine GIF](https://github.com/user-attachments/assets/60b86fca-22c5-4f30-9b28-3e3502a00c87) | ![Adjusted n_grid Plasticine GIF](https://github.com/user-attachments/assets/ae3db0fa-80fb-4fe0-bc55-cd7e3bbdb85c) |

**Note:** The baseline `n_grid` value is 100 for `sand`, and 25 for both `metal` and `plasticine`. The adjusted value is set to 50 for all materials. Hence, the adjusted `n_grid` is not always lower or higher than the baseline, but rather a comparative value to analyze the effect of grid resolution.
#### Overall Observations & Insights for `n_grid`

[Your comparative observations for the `n_grid` parameter go here.]

---

#### 2. Adjusting `substep_dt`

This study explores the effect of the simulation time step size (`substep_dt`).

* **Material:** `sand`
    * **PSNR:** 15.52

        ![PSNR vs Frame](imgs/substep_dt/sand.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/TjX9gdCpIvY?feature=share)
* **Material:** `metal`
    * **PSNR:** 18.35

        ![PSNR vs Frame](imgs/substep_dt/metal.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/apv-FjOMsZQ?feature=share)
* **Material:** `plasticine`
    * **PSNR:** 22.64

        ![PSNR vs Frame](imgs/substep_dt/plasticine.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/UkUaPRoWHSk?feature=share)

##### Visual Comparison for `substep_dt`

| Material    | Baseline (`substep_dt`: default)                 | Adjusted (`substep_dt`: 1e-5)                      |
| :---------- | :-------------------------------------------: | :-------------------------------------------: |
| **Sand** | ![Baseline Sand GIF](https://github.com/user-attachments/assets/8fcc079e-9af9-4634-b71a-0a6d60168805) | ![Adjusted substep_dt Sand GIF](https://github.com/user-attachments/assets/fb4d3824-4809-4647-96f3-2da5b074e766) |
| **Metal** | ![Baseline Metal GIF](https://github.com/user-attachments/assets/2a4f90f9-7135-4c97-92b2-ee5788c1a7a2) | ![Adjusted substep_dt Metal GIF](https://github.com/user-attachments/assets/4c68ba67-fad8-4729-950f-4bf0bb577fe0) |
| **Plasticine** | ![Baseline Plasticine GIF](https://github.com/user-attachments/assets/60b86fca-22c5-4f30-9b28-3e3502a00c87) | ![Adjusted substep_dt Plasticine GIF](https://github.com/user-attachments/assets/3e0aef98-57f9-4e48-96e7-ce874f6e9725) |

##### Overall Observations & Insights for `substep_dt`

[Your comparative observations for the `substep_dt` parameter go here.]

---

#### 3. Adjusting `grid_v_damping_scale`

This study explores the effect of the grid velocity damping factor (`grid_v_damping_scale`).

* **Material:** `sand`
    * **PSNR:** 15.33

        ![PSNR vs Frame](imgs/grid_v_damping_scale/sand.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/MOIi2vrwgAA?feature=share)
* **Material:** `metal`
    * **PSNR:** 17.67

        ![PSNR vs Frame](imgs/grid_v_damping_scale/metal.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/Vt2kfQB-NG8?feature=share)
* **Material:** `plasticine`
    * **PSNR:** 29.77

        ![PSNR vs Frame](imgs/grid_v_damping_scale/plasticine.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/mFf3deCnkcA?feature=share)

##### Visual Comparison for `grid_v_damping_scale`

| Material    | Baseline (`grid_v_damping_scale`: default)                 | Adjusted (`grid_v_damping_scale`: 0.95)                      |
| :---------- | :-------------------------------------------: | :-------------------------------------------: |
| **Sand** | ![Baseline Sand GIF](https://github.com/user-attachments/assets/8fcc079e-9af9-4634-b71a-0a6d60168805) | ![Adjusted damping Sand GIF](https://github.com/user-attachments/assets/1e9e694d-016b-4da6-aa0f-4c259578b109) |
| **Metal** | ![Baseline Metal GIF](https://github.com/user-attachments/assets/2a4f90f9-7135-4c97-92b2-ee5788c1a7a2) | ![Adjusted damping Metal GIF](https://github.com/user-attachments/assets/79baaa0b-6d0e-4746-83f4-a9cb40ac19a1) |
| **Plasticine** | ![Baseline Plasticine GIF](https://github.com/user-attachments/assets/60b86fca-22c5-4f30-9b28-3e3502a00c87) | ![Adjusted damping Plasticine GIF](https://github.com/user-attachments/assets/ae3db0fa-80fb-4fe0-bc55-cd7e3bbdb85c) |

##### Overall Observations & Insights for `grid_v_damping_scale`

[Your comparative observations for the `grid_v_damping_scale` parameter go here.]

---

#### 4. Adjusting `softening`

This study explores the effect of the stress softening factor (`softening`).

* **Material:** `sand`
    * **PSNR:** 31.93

        ![PSNR vs Frame](imgs/softening/sand.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/3DcBmdQ2S40?feature=share)
* **Material:** `metal`
    * **PSNR:** 38.52

        ![PSNR vs Frame](imgs/softening/metal.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/H9I1ZmHY5hw?feature=share)
* **Material:** `plasticine`
    * **PSNR:** 41.81

        ![PSNR vs Frame](imgs/softening/plasticine.png)
    * **Simulation Video:** [Link to video](https://youtube.com/shorts/KQwwl2Oxij8?feature=share)

##### Visual Comparison for `softening`

| Material    | Baseline (`softening`: default)                 | Adjusted (`softening`: 0.3)                      |
| :---------- | :-------------------------------------------: | :-------------------------------------------: |
| **Sand** | ![Baseline Sand GIF](https://github.com/user-attachments/assets/8fcc079e-9af9-4634-b71a-0a6d60168805) | ![Adjusted softening Sand GIF](https://github.com/user-attachments/assets/e0d3fb14-c9dc-42b4-adb2-f10bfd8f95a4) |
| **Metal** | ![Baseline Metal GIF](https://github.com/user-attachments/assets/2a4f90f9-7135-4c97-92b2-ee5788c1a7a2) | ![Adjusted softening Metal GIF](https://github.com/user-attachments/assets/a2454b96-5264-439e-8c19-e5c4bba32bb2) |
| **Plasticine** | ![Baseline Plasticine GIF](https://github.com/user-attachments/assets/60b86fca-22c5-4f30-9b28-3e3502a00c87) | ![Adjusted softening Plasticine GIF](https://github.com/user-attachments/assets/3a805f70-34e4-49f2-a25d-936eb405d1d1) |

##### Overall Observations & Insights for `softening`

[Your comparative observations for the `softening` parameter go here.]

---

### BONUS: Automatic Parameter Inference

[Your detailed answer to the bonus question goes here. You should describe a potential framework, perhaps involving machine learning, optimization, or other techniques, to address the limitation that parameters are manually defined.]

---

## Reference
```bibtex
@inproceedings{xie2024physgaussian,
    title     = {Physgaussian: Physics-integrated 3d gaussians for generative dynamics},
    author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```
