import os
from contextlib import nullcontext
import torch
try:
    from diffusers import (
        DPMSolverMultistepScheduler, 
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        AutoencoderKL, 
        UNet2DConditionModel, 
        LCMScheduler, 
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler,
        UniPCMultistepScheduler
    )
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config,
        create_text_encoder_from_ldm_clip_checkpoint
    )            
except:
    raise ImportError("Diffusers version too old. Please update to 0.26.0 minimum.")
from .scheduling_tcd import TCDScheduler
from contextlib import nullcontext
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

from .hidiffusion import apply_hidiffusion, remove_hidiffusion

from omegaconf import OmegaConf
from transformers import CLIPTokenizer
import comfy.model_management as mm
import comfy.utils
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
    
class diffusers_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("DIFFUSERSMODEL",)
    RETURN_NAMES = ("diffusers_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "IC-Light-Wrapper"

    def loadmodel(self, model, clip, vae):
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        device = mm.get_torch_device()

        custom_config = {
            'model': model,
            'vae': vae,
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(5)
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))

            print("loading ELLA")
            checkpoint_path = os.path.join(folder_paths.models_dir,'ella')
            ella_path = os.path.join(checkpoint_path, 'ella-sd1.5-tsc-t5xl.safetensors')
            if not os.path.exists(ella_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="QQGYLab/ELLA", local_dir=checkpoint_path, local_dir_use_symlinks=False)
            
            with (init_empty_weights() if is_accelerate_available() else nullcontext()):
                converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
                new_vae = AutoencoderKL(**converted_vae_config)

                converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
                unet = UNet2DConditionModel(**converted_unet_config)
                
            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)

            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            if is_accelerate_available():
                for key in converted_vae:
                    set_module_tensor_to_device(new_vae, key, device=device, dtype=dtype, value=converted_vae[key])
            else:
                new_vae.load_state_dict(converted_vae, strict=False)
            del converted_vae
            pbar.update(1)

            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            if is_accelerate_available():
                for key in converted_unet:
                    set_module_tensor_to_device(unet, key, device=device, dtype=dtype, value=converted_unet[key])
            else:
                unet.load_state_dict(converted_unet, strict=False)
            del converted_unet

            pbar.update(1)
            # 3. text_model
            print("loading text model")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1
            }
            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            scheduler=DPMSolverMultistepScheduler(**scheduler_config)
            pbar.update(1)
            del sd

            pbar.update(1)

            print("creating pipeline")
            self.pipe = StableDiffusionImg2ImgPipeline(
                unet=unet,
                vae=new_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                image_encoder=None
            )
            print("pipeline created")
            pbar.update(1)
            #self.pipe.enable_model_cpu_offload()
            diffusers_model = {
                'pipe': self.pipe,
            }
   
        return (diffusers_model,)        
        
class LoadICLightUnetDiffusers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusersmodel": ("DIFFUSERSMODEL",),
                "model_path": (folder_paths.get_filename_list("unet"), )
            } 
        }

    RETURN_TYPES = ("DIFFUSERSMODEL",)
    FUNCTION = "load"
    CATEGORY = "IC-Light-Wrapper"

    def load(self, diffusersmodel, model_path):
        unet = diffusersmodel["pipe"].unet
        device = mm.get_torch_device()

        unet_original_forward = unet.forward

        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs['cross_attention_kwargs'] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
        
        unet.forward = hooked_unet_forward

        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadICLightUnet: Loading LoadICLightUnet weights")
            from comfy.utils import load_torch_file
            sd_offset = load_torch_file(model_full_path, device=mm.get_torch_device())
            sd_origin = unet.state_dict()
            keys = sd_origin.keys()
            sd_merged = {k: sd_origin[k].to(device) + sd_offset[k].to(device) for k in sd_origin.keys()}
            unet.load_state_dict(sd_merged, strict=True)
            del sd_offset, sd_origin, sd_merged, keys

        return diffusersmodel,

class iclight_diffusers_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "diffusers_model": ("DIFFUSERSMODEL",),
            "latent": ("LATENT",),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 2.0, "min": 1.01, "max": 20.0, "step": 0.01}),
            "denoise_strength": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler',
                    'UniPCMultistepScheduler',
                    'TCDScheduler'
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
            "prompt": ("STRING", {"default": "positive", "multiline": True}),
            "n_prompt": ("STRING", {"default": "negative", "multiline": True}),
            "hidiffusion": ("BOOLEAN", {"default": False}),
            },
            "optional"  : {
                "bg_latent": ("LATENT",),
                "fixed_seed": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "IC-Light-Wrapper"

    def process(self, latent, diffusers_model, width, height, steps, guidance_scale, denoise_strength, seed, scheduler, prompt, n_prompt, hidiffusion, bg_latent=None, fixed_seed=True):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        pipe=diffusers_model['pipe']
        pipe.to(device, dtype=dtype)
        scale_factor = pipe.vae.config.scaling_factor

        scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1,
            }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == 'UniPCMultistepScheduler':
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        elif scheduler == 'TCDScheduler':
            noise_scheduler = TCDScheduler(**scheduler_config)
        
        pipe.scheduler = noise_scheduler
        if hidiffusion:
            apply_hidiffusion(pipe)
        else:
            remove_hidiffusion(pipe)

        if bg_latent is not None:
            bg_latent = bg_latent["samples"]
            bg_latent = bg_latent * pipe.vae.config.scaling_factor
        else:
            bg_latent = None

        concat_conds = latent["samples"]
        concat_conds = concat_conds * pipe.vae.config.scaling_factor
        B, H, W, C = latent["samples"].shape
        prompt_list = []
        prompt_list.append(prompt)
        if len(prompt_list) < B:
            prompt_list += [prompt_list[-1]] * (B - len(prompt_list))

        n_prompt_list = []
        n_prompt_list.append(n_prompt)
        if len(n_prompt_list) < B:

            n_prompt_list += [n_prompt_list[-1]] * (B - len(n_prompt_list))

        if fixed_seed:
            generator = [torch.Generator(device=device).manual_seed(seed) for _ in range(B)]
        else:
            generator= [torch.Generator(device="cuda").manual_seed(i) for i in range(B)]

        pbar = comfy.utils.ProgressBar(steps)
        def progress_counter_callback(pipeline, step, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs or {}

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            
            images = pipe(
            image=bg_latent,
            prompt = prompt_list,
            strength = denoise_strength,
            negative_prompt = n_prompt_list,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=guidance_scale,
            num_inference_steps=int(round(steps / denoise_strength)),
            height=height,
            width=width,
            cross_attention_kwargs={'concat_conds': concat_conds},
            generator=generator,
            output_type="latent",
            callback_on_step_end=progress_counter_callback,
            #callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"],
            ).images
            images = images / scale_factor
            #image_out = images.permute(0, 2, 3, 1).cpu().float()
            return ({"samples": images},)
                
NODE_CLASS_MAPPINGS = {
    "diffusers_model_loader": diffusers_model_loader,
    "LoadICLightUnetDiffusers": LoadICLightUnetDiffusers,
    "iclight_diffusers_sampler": iclight_diffusers_sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "diffusers_model_loader": "Diffusers Model Loader",
    "LoadICLightUnetDiffusers": "LoadICLightUnetDiffusers",
    "iclight_diffusers_sampler": "IC-Light Diffusers Sampler"
}
