import torch
from einops import rearrange, repeat
from torch import nn, einsum

from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention as CrossAttention
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
import numpy as np

class CustomDiffusion(LatentDiffusion):
    def __init__(self,
                 freeze_model='crossattn-kv',
                 cond_stage_trainable=False,
                 add_token=False,
                 opt_embed=False,
                 feature_delta=1e-2,
                 *args, **kwargs):

        self.feature_delta = feature_delta
        self.opt_embed = opt_embed
        self.freeze_model = freeze_model
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)

        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif not ('per_k' in x[0] or 'per_v' in x[0]):
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True

        def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False
                else:
                    change_checkpoint(layer)

        change_checkpoint(self.model.diffusion_model)

        feature_delta_copy = self.feature_delta
        def new_forward(self, x, context=None, mask=None):
            h = self.heads
            crossattn = False
            if context is not None:
                crossattn = True
            q = self.to_q(x)
            context = default(context, x)

            k = self.to_k(context)
            v = self.to_v(context)

            per_k = self.per_k(context)
            per_v = self.per_v(context)
            k = feature_delta_copy*per_k + k
            v = feature_delta_copy*per_v + v

            if crossattn:
                modifier = torch.ones_like(k)
                modifier[:, :1, :] = modifier[:, :1, :]*0.
                k = modifier*k + (1-modifier)*k.detach()
                v = modifier*v + (1-modifier)*v.detach()

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            del q, k
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        def change_forward(model):
            for layer in model.children():
                if type(layer) == CrossAttention:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    change_forward(layer)

        change_forward(self.model.diffusion_model)

    def configure_optimizers(self):
        lr = self.learning_rate
        embedding_lr = self.embedding_learning_rate
        params = []
        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'per_k' in x[0] or 'per_v' in x[0]:
                        params += [x[1]]
                        print(x[0])
        elif self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn2' in x[0]:
                        params += [x[1]]
                        print(x[0])
        else:
            params = list(self.model.parameters())

        if self.embedding_manager is not None: # If using textual inversion
            embedding_params = list(self.embedding_manager.embedding_parameters())

        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        if not self.opt_embed:
            print("we do NOT optimize the embedding manager.")
            opt = torch.optim.AdamW(params, lr=lr)
        else:
            if self.embedding_manager is None:
                raise ValueError("print implement the embedding manager.")
            print("we DO optimize the embedding manager.")
            opt = torch.optim.AdamW([
                                    {"params": embedding_params, "lr": embedding_lr}, 
                                    {"params": params}], lr=lr)
        return opt

    def p_losses(self, x_start, cond, t, mask=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_simple = (loss_simple*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_simple = loss_simple.mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = (self.logvar.to(self.device))[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_vlb = (loss_vlb*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_vlb = loss_vlb.mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def get_input_withmask(self, batch, **args):
        out = super().get_input(batch, self.first_stage_key, **args)
        # mask = batch["mask"]
        # if len(mask.shape) == 3:
        #     mask = mask[..., None]
        # mask = rearrange(mask, 'b h w c -> b c h w')
        # mask = mask.to(memory_format=torch.contiguous_format).float()
        # out += [mask]
        # print(out[0].size())
        # print(out[2].size())
        # print("mask: ", mask.size())
        # import sys;sys.exit(0)
        out += [None]
        return out

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            train_batch = batch[0]
            train2_batch = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_train2, _ = self.shared_step(train2_batch)
            loss = loss_train + loss_train2
        else:
            train_batch = batch
            loss, loss_dict = self.shared_step(train_batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input_withmask(batch, **kwargs)
        loss = self(x, c, mask=mask)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                unconditional_guidance_scale=6.
                unconditional_conditioning = self.get_learned_conditioning(len(c) * [""])
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                        unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples_scaled"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
            #         self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
            #                                                  ddim_steps=ddim_steps,eta=ddim_eta,
            #                                                  quantize_denoised=True)
            #         # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
            #         #                                      quantize_denoised=True)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

        return log