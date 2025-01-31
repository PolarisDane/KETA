# KETA: Kinematic-Phrases-Enhanced Text-to-Motion Generation via Fine-grained Alignment

Please visit our [**webpage**](https://polarisdane.github.io/KETA/) for more details.


## Getting started

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```

```shell
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

<details>
  <summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

</details>

### 2. Get data

<details>
  <summary><b>Text to Motion</b></summary>


#### Full data (text + motion capture)

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

Then run Kinematic Phrases Extractor script to get motions in fine-grained representation.

```shell
python ./text_2_motion/diffusion/kp_extract.py
```

</details>

## Train your own model

<details>
  <summary><b>Text to Motion</b></summary>

**HumanML3D**
```shell
python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset humanml
```

</details>

* Use `--diffusion_steps 50` to train the faster model with less diffusion steps.
* Use `--device` to define GPU id.
* Use `--arch` to choose one of the architectures reported in the paper `{trans_enc, trans_dec}` (`trans_enc` is default).
* Add `--eval_during_training` to run a short evaluation for each saved checkpoint.
  This will slow down training but will give you better monitoring.

## Evaluate

<details>
  <summary><b>Text to Motion</b></summary>


**HumanML3D**

You can download necessary dependencies from [here](https://drive.google.com/drive/folders/10s5HXSFqd6UTOkW2OMNc27KGmMLkVc2L). Thanks for this [discussion](https://github.com/GuyTevet/motion-diffusion-model/issues/222).

```shell
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
```

</details>

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model), [Kinematic Phrases](https://github.com/Foruck/Kinematic-Phrases).
