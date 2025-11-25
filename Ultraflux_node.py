 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from diffusers.hooks import apply_group_offloading
from pathlib import PureWindowsPath
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm

from .inf_ultraflux import load_dit,inference
from .model_loader_utils import phi2narry

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir



class UltraFlux_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="UltraFlux_SM_Model",
            display_name="UltraFlux_SM_Model",
            category="UltraFlux",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
                io.String.Input("repo",default=""), 
            ],
            outputs=[
                io.Custom("UltraFlux_SM_Model").Output(display_name="pipeline"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,vae,repo) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        if repo:
            repo=PureWindowsPath(repo).as_posix()
        pipeline = load_dit(dit_path,gguf_path,repo,vae_path,node_cr_path)
        return io.NodeOutput(pipeline)


class UltraFlux_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UltraFlux_SM_KSampler",
            display_name="UltraFlux_SM_KSampler",
            category="UltraFlux",
            inputs=[
                io.Custom("UltraFlux_SM_Model").Input("pipeline"),
                io.Combo.Input("lora1",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("lora2",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("lora_scale1", default=1, min=0.1, max=1.0,step=0.1, round=0.01,),
                io.Float.Input("lora_scale2", default=1, min=0.1, max=1.0,step=0.1, round=0.01,),
                io.Conditioning.Input("cond"),
                io.Int.Input("width", default=4096, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=4096, min=512, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("steps", default=8, min=1, max=1024,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=4.0, min=0, max=20,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=10, min=1, max=MAX_SEED,display_mode=io.NumberDisplay.number),
            ], # io.Float.Input("noise", default=0.0, min=0.0, max=1.0,step=0.01,display_mode=io.NumberDisplay.number),
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    @classmethod
    def execute(cls, pipeline,lora1,lora2,lora_scale1,lora_scale2,cond,width,height,steps,guidance_scale,seed,block_num,) -> io.NodeOutput:

        cf_models=mm.loaded_models()
        try:
            for pipe in cf_models:   
                pipe.unpatch_model(device_to=torch.device("cpu"))
                print(f"Unpatching models.{pipe}")
        except: pass
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        max_gpu_memory = torch.cuda.max_memory_allocated()
        print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
        
        lora_path1=folder_paths.get_full_path("loras", lora1) if lora1 != "none" else None
        lora_path2=folder_paths.get_full_path("loras", lora2) if lora2 != "none" else None
        lora_list=[i for i in [lora_path1,lora_path2] if i is not None]
    
        lora_scales=[lora_scale1,lora_scale2]
        if lora_list:
            if len(lora_list)!=len(lora_scales): #sacles  
                lora_scales = lora_scales[:1]
            all_adapters = pipeline.get_list_adapters()
            dit_list=[]
            if all_adapters:
                dit_list= all_adapters['transformer']
            adapter_name_list=[]
            for path in lora_list:
                if path is not None:
                    name=os.path.splitext(os.path.basename(path))[0].replace(".", "_")
                    adapter_name_list.append(name)
                    if name in dit_list:
                        continue
                    pipeline.load_lora_weights(path, adapter_name=name)
            print(f"成功加载LoRA权重: {adapter_name_list} (scale: {lora_scales})")        
            pipeline.set_adapters(adapter_name_list, adapter_weights=lora_scales)
            try:
                active_adapters = pipeline.get_active_adapters()
                all_adapters = pipeline.get_list_adapters()
                print(f"当前激活的适配器: {active_adapters}")
                print(f"所有可用适配器: {all_adapters}") 
            except:
                pass
        try:
            dit_list= pipeline.get_list_adapters()['transformer']
            for name in dit_list:
                if lora_list :
                    name_list=[os.path.splitext(os.path.basename(i))[0].replace(".", "_") for i in lora_list ]
                    if name in name_list: #dit_list
                        continue
                    else:
                        pipeline.delete_adapters(name)
                        print(f"去除dit中未加载的lora: {name}")  
                else:
                    pipeline.delete_adapters(name)
        except:pass
            
        # apply offloading
        apply_group_offloading(pipeline.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        # infer
        image=inference(pipeline,cond,guidance_scale,steps,seed,width,height)
 
        return io.NodeOutput(phi2narry(image))

from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/UltraFlux_SM_Extension")
async def get_hello(request):
    return web.json_response("UltraFlux_SM_Extension")

class UltraFlux_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UltraFlux_SM_Model,
            UltraFlux_SM_KSampler,

        ]
async def comfy_entrypoint() -> UltraFlux_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return UltraFlux_SM_Extension()



