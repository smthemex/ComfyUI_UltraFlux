# ComfyUI_UltraFlux
[UltraFlux](https://github.com/W2GenAI-Lab/UltraFlux):Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios,try it in comfyUI


# Coming soon


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_UltraFlux

```

2.requirements  
----
```
pip install -r requirements.txt
```

3.Model
----

* gguf [links](https://huggingface.co/smthem/UltraFlux-v1-gguf/tree/main) optional/备选
* vae ranmae it  [v1](https://huggingface.co/Owen777/UltraFlux-v1) 
* diffuser transformer   [v1](https://huggingface.co/Owen777/UltraFlux-v1) or [v1.1](https://huggingface.co/Owen777/UltraFlux-v1-1-Transformer) optional/备选 填repo
* comfyUI normal T5 and clip-l 
* lora, anyturbo and style flux lora #任意flux加速和风格lora
```
├── ComfyUI/models/gguf # or fp8 
|     ├── UltraFlux-v1-1-BF16.gguf # or Q8
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
![](https://github.com/smthemex/ComfyUI_UltraFlux/blob/main/example_workflows/example.png)

5.Citation
----
```
--
```
