import random
import numpy as np
import torch
from pytorch_lightning import seed_everything
import einops
import sys 
import os 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from config import *


def test_sample(model,ddim_sampler):
    with torch.no_grad():
        prompt="a cat in a rowboat"
        n_prompt="dog"
        num_samples=1
        ddim_steps=40
        seed=-1
        strength=1.0
        scale=9.0
        img = np.random.randint(0, 256, size=(768, 768, 3), dtype=np.uint8)
        H, W, C = img.shape
        guess_mode=False
        eta=0.0

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - img] + results

def test():
    print("Creating model...")
    model=create_model(MODEL_CONFIG_LOCAL).cpu()
    #move the model to GPU
    model = model.cuda()
    #instantiate the sampler with the gpu-located model
    sampler = DDIMSampler(model)
    print("Test uninitialized model sample...")
    test_sample(model,sampler)
    print("loading model...")
    m,u=model.load_state_dict(load_state_dict(MODEL_LOCAL, location=model.device), strict=False)
    print("Test with initialized model...")
    test_sample(model,sampler)

if __name__ == "__main__":
    print("testing sampler...")
    test()

