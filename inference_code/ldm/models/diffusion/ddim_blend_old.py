"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               ref_img_feature=None,
               init_image=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               mask_orig=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        print("in the sample, is none or not: ", mask_orig is None)
        # print("inside the ddim here.")
        # print(ref_img_feature.size())
        # import sys;sys.exit(0)
        samples, intermediates = self.ddim_sampling(conditioning, size, ref_img_feature,
                                                    init_image=init_image,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0, mask_orig=mask_orig,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, ref_img_feature, init_image=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, mask_orig=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]


        
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        ## adding it here.
        if init_image is not None:
            assert (
                x0 is None and x_T is None
            ), "Try to infer x0 and x_t from init_image, but they already provided"

            encoder_posterior = self.model.encode_first_stage(init_image)
            x0 = self.model.get_first_stage_encoding(encoder_posterior)
            last_ts = torch.full((1,), time_range[0], device=device, dtype=torch.long)
            # print("time_range: ", time_range)   # [991 981 971 961 951 941
            # print(last_ts)  # tensor([991], device='cuda:0')
            # import sys;sys.exit(0) x_T is the gaussian noise here.
            x_T = torch.cat([self.model.q_sample(x0, last_ts) for _ in range(b)])
            img = x_T
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        percentage_of_pixel_blending = 0
        cutoff_point = int(len(time_range) * (1 - percentage_of_pixel_blending))
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # print("skip this mask update.")
            # if mask is not None:
            #     assert x0 is not None
            #     img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            #     img = img_orig * mask + (1. - mask) * img
            # Latent-level blending
            # if mask is not None and i < cutoff_point:
            # if mask is not None and (5 < i and i < 15):
                # print("doing the blending here.")
            # if mask is not None and 5 < i:
            # if mask is not None and i < 10:
            if mask is not None:
                # print("inside latent ", i, cutoff_point)
                n_masks = mask.shape[0]
                masks_interval = len(time_range) // n_masks + 1
                curr_mask = mask[i // masks_interval].unsqueeze(0)
                img_orig = self.model.q_sample(x0, ts)
                
                ## img is the current ((t-1)-th) step latent images, diffused by x_T
                ## img_orig is the current ((t-1)-th) step latent images, predicted by x_0

                # print("img_orig: ", x0.size(), img_orig.size())
                # img:  torch.Size([4, 4, 32, 32]) torch.Size([4, 4, 32, 32])
                # if i < 18:
                img = img_orig * (1 - curr_mask) + curr_mask*img
                # else:
                    # img = img_orig * (1 - curr_mask*0.5) + curr_mask*0.5*img
                    
                # print("img: ", img.size(), img_orig.size())
                # torch.Size([4, 4, 32, 32]) torch.Size([4, 4, 32, 32])
                # print("inside the latent-level blending.")
                # import sys;sys.exit(0)

            # Pixel-level blending
            # your function never enters into the pixel-level blending here.
            # print("is none or not: ", mask_orig is None)
            # import sys;sys.exit(0)
            # if mask_orig is not None and 10 > i and i >= 5:
            #     print("inside the pixel. ", i, cutoff_point)
            #     foreground_pixels = self.model.decode_first_stage(pred_x0)
            #     background_pixels = init_image
            #     pixel_blended = foreground_pixels * mask_orig + background_pixels * (1 - mask_orig)
            #     img_x0 = self.model.get_first_stage_encoding(
            #         self.model.encode_first_stage(pixel_blended)
            #     )
            #     img = self.model.q_sample(img_x0, ts)

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, ref_img_feature, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, ref_img_feature,
                      index=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        # print("inside the p_sample_ddim.")
        # print("ref_img_feature: ", ref_img_feature.size())
        # import sys;sys.exit(0)

        # print("x is: ", x.size())   # torch.Size([2, 4, 32, 32])
        # print("c c_crossattn is: ", torch.cat(c['c_crossattn'], 1).size())  # torch.Size([2, 77, 768])
        # print("c c_concat is: ", torch.cat(c['c_concat'], 1).size())    # torch.Size([2, 3, 256, 256])
        # print("t is: ", t.size())   # torch.Size([2])
        # model_output = self.model.apply_model(x, t, c, ref_img_feature)
        # print("output: ", model_output.size())  # torch.Size([2, 4, 32, 32])
        # import sys;sys.exit(0)
        # print("inside the core p_sample_ddim here.")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output, _, _ = self.model.apply_model(x, t, c, ref_img_feature)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            ref_img_feature_in = torch.cat([ref_img_feature] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # print("inside here...")
            # print("x is: ", x_in.size())   # torch.Size([2, 4, 32, 32])
            # print("c c_crossattn is: ", torch.cat(c_in['c_crossattn'], 1).size())  # torch.Size([2, 77, 768])
            # print("c c_concat is: ", torch.cat(c_in['c_concat'], 1).size())    # torch.Size([2, 3, 256, 256])
            # print("t is: ", t_in.size())   # torch.Size([2])
            # model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, ref_img_feature_in).chunk(2)
            # print("model_uncond: ", model_uncond.size())    # model_uncond:  torch.Size([2, 4, 32, 32])
            # print("model_t: ", model_t.size())  # model_t:  torch.Size([2, 4, 32, 32])
            # import sys;sys.exit(0)
            model_eps, model_ld_feat, model_pm_feat = self.model.apply_model(x_in, t_in, c_in, ref_img_feature)
            model_uncond, model_t = model_eps.chunk(2)
            # model_uncond, model_t, _ = self.model.apply_model(x_in, t_in, c_in, ref_img_feature_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0