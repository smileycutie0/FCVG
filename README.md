# Generative Inbetweening through Frame-wise Conditions-Driven Video Generation
#### Tianyi Zhu,  Dongwei Ren, Qilong Wang, Xiaohe Wu, Wangmeng Zuo
This repository is the official PyTorch implementation of "Generative Inbetweening through Frame-wise Conditions-Driven Video Generation".

[![arXiv](https://img.shields.io/badge/arXiv-2412.xxxx-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://fcvg-inbetween.github.io/)

## 🖼️ Resluts

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Inbetweening results</td>
    </tr>
  <tr>
  <td>
    <img src=example/real/003/00.png width="250">
  </td>
  <td>
    <img src=example/real/003/24.png width="250">
  </td>
  <td>
    <img src=example/real/003/out.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=example/real/002/00.png width="250">
  </td>
  <td>
    <img src=example/real/002/24.png width="250">
  </td>
  <td>
    <img src=example/real/002/out.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=example/animation/003/00.jpg width="250">
  </td>
  <td>
    <img src=example/animation/003/24.jpg width="250">
  </td>
  <td>
    <img src=example/animation/003/out.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=example/animation/002/00.png width="250">
  </td>
  <td>
    <img src=example/animation/002/24.png width="250">
  </td>
  <td>
    <img src=example/animation/002/out.gif width="250">
  </td>
  </tr> 
</table>



## ⚙️ Run inference demo
#### 1. Setup environment

```shell
git clone https://github.com/Tian-one/FCVG.git
cd FCVG
```

```
conda create -n FCVG python=3.10.14
conda activate FCVG
pip install -r requirements.txt
```

#### 2. Download models

1. Download the [Gluestick](https://github.com/cvg/GlueStick) weights and put them in './models/resources'.

   ```
   wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar -P models/resources/weights
   ```

2. Download the  [DWPose](https://github.com/IDEA-Research/DWPose) pretrained weights dw-ll_ucoco_384.onnx and yolox_l.onnx [here](https://drive.google.com/drive/folders/1Ftv-jR4R8VtnOyy38EVLRa0yLz0-BnUY?usp=sharing), then put them in './checkpoints/dwpose'. 

3. Download our FCVG model [here](https://drive.google.com/drive/folders/1qIvr9WO8qk3NUdztxweTmexfkHt8oRDB?usp=sharing), put them in './checkpoints'

#### 3. Run the inference script

Run inference with default setting:

``` shell
bash demo.sh
```

or run

```
python demo_FCVG.py 
```

>   --pretrained_model_name_or_path: pretrained SVD model folder, we fintune models based on [SVD-XT1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
>   --controlnext_path:  [ControlNeXt](https://github.com/dvlab-research/ControlNeXt) model path
>   --unet_path: finetuned unet model path
>   --image1_path: start frame path
>   --image2_path: end frame path
>   --output_dir: folder path to save the results
>   --control_weight: frame-wise condition control weight, default is 1.0
>   --num_inference_steps: diffusion denoise steps, default is 25
>   --height : input frames height, default is 576
>   --width: input frames width, default is 1024



## ✨ News/TODO

- [x] Inference code of FCVG
- [ ] Release  Datasets



## 🖊️ Citation

```bibtex

```



## 💞 Acknowledgements

Thanks for the work of [ControlNeXt](https://github.com/dvlab-research/ControlNeXt), [svd_keyframe_interpolation](https://github.com/jeanne-wang/svd_keyframe_interpolation), [GlueStick](https://github.com/cvg/GlueStick), [DWPose](https://github.com/IDEA-Research/DWPose). Our code is based on the implementation of them.