# ComfyUI_UltraFlux
[UltraFlux](https://github.com/W2GenAI-Lab/UltraFlux):Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios,try it in comfyUI

# Update
* 图生图模式上线，加噪伪超分 / i2i is done。
* 因为基于flux ，如果出现人物，推荐使用修手lora，风格lora因为微调图片精度不够，可能会劣化输出，8G显存block number适当从10下调，4G显存，你就从1开始往上测试吧
* Because based on flux, if a character appears, it is recommended to use a hand fix Lora. The style Lora may degrade the output due to insufficient fine-tuning of the image accuracy. The block number of 8GB VRAM should be appropriately reduced from 10, and 4G VRAM should be tested from 1 onwards


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_UltraFlux

```
2.requirements  i
----
* 不装也行，没什么需求
```
pip install -r requirements.txt
```
3.Model
----

* gguf or transformer [smthem/UltraFlux-v1-gguf](https://huggingface.co/smthem/UltraFlux-v1-gguf/tree/main) optional/推荐用fp16 ,因为块卸载,只要内存大 
* vae rnamae it  [v1](https://huggingface.co/Owen777/UltraFlux-v1)   
* diffusers transformer   [v1](https://huggingface.co/Owen777/UltraFlux-v1) or [v1.1](https://huggingface.co/Owen777/UltraFlux-v1-1-Transformer) optional/备选 填repo的方式，一般不用    
* comfyUI normal T5 and clip-l 
* lora, any turbo and style flux lora #任意flux加速和风格lora，部分Lora的精度不够 可能会劣化输出  
```
├── ComfyUI/models/gguf # or transformer
|     ├── UltraFlux-v1-1-BF16.gguf # or Q8
├── ComfyUI/models/diffusion_models # or gguf
|     ├── UltraFlux-v1-1-BF16..safetensors # or e4m3fn
├── ComfyUI/models/vae
|        ├─diffusion_pytorch_model.safetensors  # rename it 换个名字
├── ComfyUI/models/clip
|        ├──t5xxl_fp8_e4m3fn.safetensors
|        ├──clip_l.safetensors 
├── ComfyUI/models/loras 
|        ├──any turbo lora
|        ├──any style lora

```

4.Example
----
![](https://github.com/smthemex/ComfyUI_UltraFlux/blob/main/example_workflows/example_i.png)

![](https://github.com/smthemex/ComfyUI_UltraFlux/blob/main/example_workflows/example.png)

5.Citation
----
```
@misc{ye2025ultrafluxdatamodelcodesignhighquality,
      title={UltraFlux: Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios}, 
      author={Tian Ye and Song Fei and Lei Zhu},
      year={2025},
      eprint={2511.18050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18050}, 
}
```
