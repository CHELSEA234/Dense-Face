import torch
import einops
from einops import rearrange, repeat
from torch import nn, einsum

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
# from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention, InstanceNorm, ApplyStyle
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddim_new import DDIMSampler

from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
import numpy as np
import torchvision.transforms as transforms
import cv2

## from the textual inversion code.
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class AdaIN_Block(nn.Module):
    '''
        AdaIN_block to add the arcface feature.
    '''
    def __init__(self, inner_dim, out_dim=128, latent_dim=512):
        super().__init__()
        self.inner_dim = inner_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(self.inner_dim, self.out_dim, kernel_size=3, padding=1)
        self.style_0 = ApplyStyle(latent_dim, self.inner_dim)
        self.style_1 = ApplyStyle(latent_dim, self.out_dim)
        self.instance_norm = InstanceNorm()
        self.activation = nn.ReLU()

    def forward(self, x, facenet):
        y = self.conv0(x)
        y = self.instance_norm(y.clone())
        y = self.style_0(y, facenet)
        y = self.activation(y)
        y = self.conv1(y)
        y = self.style_1(y, facenet)
        return y

class ControlledUnetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, control=None, control_id=None, 
                only_mid_control=False, 
                ref_img_feat=None,
                out_block_idx=[2, 5, 8, 11]):
        '''
            params: 
                out_block_idx: which block to add the id feature.
            reference:
                the U-Net feature map shape at different layers.
                    middle layer:  torch.Size([2, 1280, 4, 4]),  torch.Size([2, 1280, 4, 4])
                    0 torch.Size([2, 1280, 4, 4]) torch.Size([2, 1280, 4, 4])
                    1 torch.Size([2, 1280, 4, 4]) torch.Size([2, 1280, 4, 4])
                    2 torch.Size([2, 1280, 4, 4]) torch.Size([2, 1280, 4, 4]) 
                    3 torch.Size([2, 1280, 8, 8]) torch.Size([2, 1280, 8, 8])
                    4 torch.Size([2, 1280, 8, 8]) torch.Size([2, 1280, 8, 8])
                    5 torch.Size([2, 1280, 8, 8]) torch.Size([2, 640, 8, 8])  
                    6 torch.Size([2, 1280, 16, 16]) torch.Size([2, 640, 16, 16])
                    7 torch.Size([2, 640, 16, 16]) torch.Size([2, 640, 16, 16])
                    8 torch.Size([2, 640, 16, 16]) torch.Size([2, 320, 16, 16]) 
                    9 torch.Size([2, 640, 32, 32]) torch.Size([2, 320, 32, 32])
                    10 torch.Size([2, 320, 32, 32]) torch.Size([2, 320, 32, 32])
                    11 torch.Size([2, 320, 32, 32]) torch.Size([2, 320, 32, 32])  
        '''
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for idx, module in enumerate(self.input_blocks):
                h = module(h, emb, context, ref_img_feat)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            # h = module(h, emb, context)
            h = module(h, emb, context, ref_img_feat)
        h = h.type(x.dtype)
        return self.out(h)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

class ControlNet_ID(ControlNet):
    '''
        add the adain-resnet to process the facenet feature.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adain_block_3 = AdaIN_Block(1280, 1280)
        self.adain_block_3 = AdaIN_Block(1280, 1280)
        self.adain_block_2 = AdaIN_Block(640, 640)
        self.adain_block_1 = AdaIN_Block(320, 320)
        self.adain_block_0 = AdaIN_Block(320, 320)
        del self.input_hint_block
        del self.zero_convs         # you should remove all zero_convs.
        del self.middle_block_out   # the another zero_convs used in the self.middle_block.

    def forward(self, x, timesteps, context, ref_img_feat, out_indices=[0,3,6,9], **kwargs):
        '''
            currently, the output is a list of U-Net features with the identity representation.
            the indices and feature shape are: 
                0 torch.Size([2, 320, 32, 32])
                1 torch.Size([2, 320, 16, 16])
                2 torch.Size([2, 640, 8, 8])
                3 torch.Size([2, 1280, 4, 4])
                4 torch.Size([2, 1280, 4, 4])
            TODO: update this block.
        '''
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = None
        outs_id = []
        h = x.type(self.dtype)
        for idx, module in enumerate(self.input_blocks):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
                if idx == out_indices[0]:
                    h_id = self.adain_block_0(h, ref_img_feat)
                    outs_id.append(h_id)
                elif idx == out_indices[1]:
                    h_id = self.adain_block_1(h, ref_img_feat)
                    outs_id.append(h_id)
                elif idx == out_indices[2]:
                    h_id = self.adain_block_2(h, ref_img_feat)
                    outs_id.append(h_id)
                elif idx == out_indices[3]:
                    h_id = self.adain_block_3(h, ref_img_feat)
                    outs_id.append(h_id)

        h = self.middle_block(h, emb, context)
        h_id = self.adain_block_3(h, ref_img_feat)
        outs_id.append(h_id)

        return outs_id

class arcface_TI_KV_control_ld_p_mask(LatentDiffusion):

    '''
        loading data batch for the training. the data flows as follows:
            condition model: DDPM.instantiate_cond_stage: ldm.modules.encoders.modules.FrozenCLIPEmbedder
            first stage model: DDPM.instantiate_first_stage: ldm.models.autoencoder.AutoencoderKL
            control branch: arcface_latent_diffusion_control.ControlNet
                            arcface_TI_KV_control.instantiate_from_config(control_stage_config)
            UNet: arcface_latent_diffusion_control.ControlledUnetModel
            training_step
                shared_step
                    get_control_input/ LatentDiffusion.get_input ==> c is the lst of sentences.
                    forward 
                        arcface_TI_KV_control.get_learned_conditioning(c, ref_img) 
                            ==> c['c_concat'] is the lst of control_image feature.
                            ==> c['c_crossattn'] is the lst of caption, which will becomes lst of caption feature.
                            self.cond_stage_model.encode(c, ref_img, embedding_manager)
                            ==> FrozenCLIPEmbedder.encode(c, ref_img, embedding_manager)
                                FrozenCLIPEmbedder.transformer(XXXX)
                                ==> this transformer has modified functions:
                                    forward, text_model.forward, text_model.encoder.forward, and text_model.embeddings.forward
                                    text_model.embeddings.forward:
                                        embedding_manager modifies the forward the function.
                                        ==> forward function has called the facenet here.
                        p_losses (return a dict of loss)
                            apply_model
                            ==> self.control_model
    '''

    def __init__(self, 
                id_stage_config, 
                control_stage_config,
                control_id_stage_config,
                control_key, 
                freeze_model='crossattn-kv',
                cond_stage_trainable=False,
                add_token=False,
                opt_embed=False,
                feature_delta=1e-2,
                *args, **kwargs):

        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)

        ## set up the control net, target: cldm.cldm.ControlNet
        self.control_model = instantiate_from_config(control_stage_config)
        # self.control_model_ID = instantiate_from_config(control_id_stage_config)
        self.control_key = control_key
        self.only_mid_control = False
        self.control_scales = [1.0] * 13
        self.tensor_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        ## set up the facenet and hyperparameters.
        self.feature_delta = feature_delta
        self.opt_embed = opt_embed
        self.freeze_model = freeze_model
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        id_loss_func = instantiate_from_config(id_stage_config)
        id_loss_func.freeze() # keep change the model again.

        ## ugly code to brutally change the forward function in self.embedding.
        def new_forward(self, reference_img, tokenized_text, embedded_text):
            b, n, device = *tokenized_text.shape, tokenized_text.device
            if isinstance(reference_img, str):
            # if reference_img is not None:
                uncond = True   # uncond, input is a empty string.
            else:
                face_feat = reference_img
                uncond = False

            id_loss_func.facenet.eval()
            id_loss_func.facenet.training = False
            id_loss_func.eval()
            id_loss_func.training = False

            if not uncond:
                face_feat = self.projector(face_feat)
                split_flag = False
                if face_feat.size()[0] == 1:
                    split_flag = True
                    face_feat = torch.cat([face_feat, face_feat], dim=0)
                face_feat = self.projector_norm(face_feat)
                if split_flag:
                    face_feat = torch.unsqueeze(face_feat[0], dim=0)
                face_feat = self.reference_delta*face_feat

            for placeholder_string, placeholder_token in self.string_to_token_dict.items():
                placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)  
                # placeholder_idx: first element is batch_idx, second is id idx in the sample.
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding
                if len(placeholder_idx[0]) != 0 and not uncond:
                    embedded_text[placeholder_idx] = embedded_text[placeholder_idx] + face_feat[placeholder_idx[0]]
                else:
                    embedded_text[placeholder_idx] = embedded_text[placeholder_idx]
            return embedded_text
        self.embedding_manager.forward = new_forward.__get__(self.embedding_manager)
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = False
        self.id_loss_func = id_loss_func

    def apply_model(self, x_noisy, t, cond, ref_img_feat, *args, **kwargs):
        assert isinstance(cond, dict)
        ####################################################
        ## go to the LatentDiffusion to find the self.model.
        ref_img_feat = ref_img_feat.clone().detach()
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        hint = torch.cat(cond['c_concat'], 1)
        control = self.control_model(
                                    x=x_noisy,
                                    hint=hint, 
                                    timesteps=t, 
                                    context=cond_txt
                                    )
        eps = diffusion_model(
                            x=x_noisy, 
                            timesteps=t, 
                            context=cond_txt, 
                            control=control, 
                            control_id=None,
                            ref_img_feat=ref_img_feat,
                            only_mid_control=self.only_mid_control
                            ) 
        return eps

    def to_RGB(
                self, cond, ref_img, ref_img_input, target_img_bbox, 
                ddim_steps=20, ddim_eta=0.0, unconditional_guidance_scale=3.0,
                use_ddim=True, dump=False, save_name='demo.png',
                control_transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])
                ):
        N = ref_img.shape[0]
        c_cat, c = cond["c_concat"][0][:N], cond["c_crossattn"][0][:N]
        ref_ids = einops.rearrange(ref_img_input, 'b h w c -> b c h w').contiguous()
        uc = self.get_learned_conditioning(N * [""], ref_img_feature="Dummy")
        samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]}, 
                                         batch_size=N, ddim=use_ddim,
                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning={"c_concat": [c_cat], "c_crossattn": [uc]},
                                         )
        samples_cfg = self.decode_first_stage(samples_cfg)
        ref_imgs = einops.rearrange(ref_img, 'b h w c -> b c h w').contiguous()

        crop_samples_cfg = []
        for i in range(N):
            # print(samples_cfg[i,:,:,:].size())
            crop_tnr = self._crop_tensor(samples_cfg[i,:,:,:], target_img_bbox[i])
            crop_tnr = torch.unsqueeze(crop_tnr, axis=0)
            crop_samples_cfg.append(crop_tnr)
        crop_samples_cfg = torch.cat(crop_samples_cfg, axis=0)

        ## debug purpose:
        if dump:
            '''
                # point 1:
                # samples_cfg and ref_imgs are in range [0, 1] and RGB order that can be fed to facenet.                
                im = Image.fromarray(np.uint8(img))
                im.save(f"{i}_demo_check.png")
                # point 2:
                target bounding box is aligned with control mask.
                control mask is roughly aligned with generated image.
                we use target bbox to crop the generated image.
            '''
            from PIL import Image
            import cv2
            samples_cfg_ = \
                (einops.rearrange(samples_cfg, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            crop_samples_cfg_ = \
                (einops.rearrange(crop_samples_cfg, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            ref_imgs_ = \
                (einops.rearrange(ref_imgs, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            c_cat_ = \
                (einops.rearrange(c_cat, 'b c h w -> b h w c') * 255.0).cpu().numpy().clip(0, 255).astype(np.uint8)
            c_cat_ = (samples_cfg_*0.8 + c_cat_*0.2).clip(0, 255).astype(np.uint8)
            
            for i in range(N):
                cv2.imwrite(f"{i}_{save_name}", 
                                cv2.cvtColor(samples_cfg_[i], cv2.COLOR_BGR2RGB))

                cv2.imwrite(f"{i}_{save_name}".replace('.png', '_refer.png'), 
                                cv2.cvtColor(ref_imgs_[i], cv2.COLOR_BGR2RGB))

                cv2.imwrite(f"{i}_{save_name}".replace('.png', '_control.png'),
                                cv2.cvtColor(c_cat_[i], cv2.COLOR_BGR2RGB))

                cv2.imwrite(f"{i}_{save_name}".replace('.png', '_crop.png'), 
                                cv2.cvtColor(crop_samples_cfg_[i], cv2.COLOR_BGR2RGB))

        return crop_samples_cfg, ref_imgs

    def p_losses(self, x_start, cond, t, ref_img, ref_img_input, target_img_bbox, loss_flag, 
                mask=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, ref_img_feat=ref_img)

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
    def get_control_input(self, batch, k, bs=None, ref_img_feature=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, ref_img_feature=ref_img_feature, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')    # torch.Size([4, 3, 256, 256])
        control = control.to(memory_format=torch.contiguous_format).float()
        # print(c)    ['photo of a * person. a woman with long hair']
        # print(control.size())   # torch.Size([1, 3, 256, 256])
        return x, dict(c_crossattn=[c], c_concat=[control])

    def shared_step(self, batch, **kwargs):
        target_img_bbox = batch['bbox_t']
        loss_flag = batch['flag']
        ref_img = batch['refer']
        ref_img_input = torch.add(batch['refer_img'], 1.0)/2.0
        x, c = self.get_control_input(batch, self.first_stage_key, ref_img_feature=ref_img, **kwargs)
        loss = self(x, c, ref_img, ref_img_input, target_img_bbox, loss_flag)
        return loss

    def forward(self, x, c, ref_img, ref_img_input, target_img_bbox, loss_flag, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c, ref_img)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, ref_img, ref_img_input, target_img_bbox, loss_flag, *args, **kwargs)

    def get_learned_conditioning(self, c, ref_img_feature):
        '''
            by default, the c is the caption or control image.
            in our work, x is the reference image or the facenet embedding.
            embedding_manager is the class to replace text embedding with facenet feature.
            this function is called in many cases.
        '''
        if isinstance(c, dict):
            c['c_crossattn'][0] = self.cond_stage_model.encode(c['c_crossattn'][0], 
                                                                ref_img_feature, 
                                                                embedding_manager=self.embedding_manager
                                                                )
            return c
        elif isinstance(c, list):
            # print("inside the get_learned_conditioning's list condition.")
            c = self.cond_stage_model.encode(c, 
                                            ref_img_feature, 
                                            embedding_manager=self.embedding_manager
                                            )
            return c

    def training_step(self, batch, batch_idx):
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

    def configure_optimizers_embed(self):
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = False

        ## TODO: should also set controlNet as frozen?
        embedding_params = []
        for x in self.embedding_manager.named_parameters():
            if x[1].requires_grad:
                embedding_params += [x[1]]
                print(x[0])

        lr = self.embedding_learning_rate
        opt = torch.optim.AdamW(embedding_params, lr=lr)
        return opt

    def _adding_per_kv(self):
        '''
            adding feature-level perturbations.
        '''
        feature_delta_copy = self.feature_delta
        def new_forward(self, x, context=None, mask=None, facenet_feat=None):
            if facenet_feat is not None:
                # print(facenet_feat.size())      # torch.Size([2, 512])
                facenet_feat = torch.unsqueeze(facenet_feat, 1)
                # print(facenet_feat.size())      # torch.Size([2, 1, 512])
                output_feature = self.feat_to_q(facenet_feat)
                # print(output_feature.size())    # torch.Size([2, 1, 320])
                _, dim, _ = x.shape
                output_feature = output_feature.repeat(1, dim, 1)
            else:
                output_feature = x  

            h = self.heads
            crossattn = False
            if context is not None:
                crossattn = True
            q = self.to_q(x)
            per_q = self.per_q(output_feature)
            context = default(context, x)

            k = self.to_k(context)
            v = self.to_v(context)
            per_k = self.per_k(context)
            per_v = self.per_v(context)
            
            k = feature_delta_copy*per_k + k
            v = feature_delta_copy*per_v + v
            q = feature_delta_copy*per_q + q

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

    def configure_optimizers_KV(self):
        print("=====================================")
        print("Now train embedding and per_k as well as per_v layers.")
        print("=====================================")

        ## swith from cross attention to the new version.
        self._adding_per_kv()

        ## disable cond_stage_model
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        # better to double check whether self.id_loss_func is locked.
        self.id_loss_func.train = disabled_train
        self.id_loss_func.face_pool.train = disabled_train
        self.id_loss_func.facenet.train = disabled_train
        self.id_loss_func.eval()
        self.id_loss_func.face_pool.eval()
        self.id_loss_func.facenet.eval()

        ## disable the kv.
        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'adain_block_' in x[0]:
                    x[1].requires_grad = True
                    # print(x[0])
                else:
                    if 'transformer_blocks' not in x[0]:
                        x[1].requires_grad = False
                    elif not ('per_k' in x[0] or 'per_v' in x[0] or 'per_q' in x[0] or 'feat_to_q' in x[0]):
                        x[1].requires_grad = False
                    else:
                        x[1].requires_grad = True

        ## only train per_k and per_v:
        lr = self.learning_rate
        params = []
        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'per_k' in x[0] or 'per_v' in x[0] or 'per_q' in x[0] or 'feat_to_q' in x[0]:
                        params += [x[1]]
                        # print(x[0])
                elif 'adain_block_' in x[0]:
                    params += [x[1]]
                    # print(x[0])

        params += list(self.control_model.parameters())
        # params += list(self.control_model_ID.parameters())

        ## disable embedding param.
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = False

        for x in self.embedding_manager.named_parameters():
            if x[1].requires_grad:
                # embedding_params += [x[1]]
                params += [x[1]]
                # print(x[0])

        return torch.optim.AdamW(params, lr=lr)

    ## this sample_log is used to log_images, which uses sampling function from DDIM.py not DDIM_hacked.py
    @torch.no_grad()
    def sample_log(self, ref_img_feature, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, 
                                                     batch_size, 
                                                     shape, 
                                                     cond, 
                                                     ref_img_feature, 
                                                     verbose=False, 
                                                     **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        target_img_bbox = batch['bbox_t']
        ref_img_feature = batch['refer_img']
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, ref_img_feature=batch['refer'])
        
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        refer_ids = einops.rearrange(ref_img_feature, 'b h w c -> b c h w').contiguous()
        log["id"] = refer_ids[:N]  # range [-1, 1] for the viz.
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        ## get the control input.
        N = min(x.shape[0], N)
        control = batch[self.control_key]
        if N is not None:
            control = control[:N]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        c_cat = control.to(memory_format=torch.contiguous_format).float()
        log["control"] = c_cat * 2.0 - 1.0  # turn to range [-1, 1] for the viz.

        # get denoise row
        with self.ema_scope("Plotting"):
            unconditional_guidance_scale=3.
            uc = self.get_learned_conditioning(N * [""], ref_img_feature="Dummy")
            samples, z_denoise_row = self.sample_log(
                                                    ref_img_feature=batch['refer'][:N], # GX: ugly code here.
                                                    cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                    batch_size=N,ddim=use_ddim,
                                                    ddim_steps=ddim_steps,eta=ddim_eta,
                                                    unconditional_conditioning={"c_concat": [c_cat], "c_crossattn": [uc]},
                                                    unconditional_guidance_scale=unconditional_guidance_scale
                                                    )
        samples_cfg = self.decode_first_stage(samples)
        log["samples_scaled"] = samples_cfg

        return log

    def _resize(self, image):
        return cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)

    def _crop_img(self, image, bbox):
        '''
            cropping the image based on the input bouning box.
        '''
        bbox = bbox[1:-1].split(',')

        x1 = int(float(bbox[0].strip())*image.shape[1])
        y1 = int(float(bbox[1].strip())*image.shape[0])
        x2 = int(float(bbox[2].strip())*image.shape[1])
        y2 = int(float(bbox[3].strip())*image.shape[0])

        if x1 <= 0:
            x1 = 0
        if y1 <= 0:
            y1 = 0
        if x2 >= image.shape[1]:
            x2 = image.shape[1]
        if y2 >= image.shape[0]:
            y2 = image.shape[0]

        # cv2.imwrite("demo.png", image)
        crop_img = image[y1:y2, x1:x2] # are you sure it is not the another way around????
        # print(image.shape, crop_img.shape, bbox)
        if crop_img.shape[0] != 256 or crop_img.shape[1] != 256:
            crop_img = self._resize(crop_img)        
        return crop_img

    def _crop_tensor(self, tensor, bbox):
        '''
            cropping the tensor based on the input bouning box.
        '''
        bbox = bbox[1:-1].split(',')
        width = tensor.size()[1]

        x1 = int(float(bbox[0].strip())*width)
        y1 = int(float(bbox[1].strip())*width)
        x2 = int(float(bbox[2].strip())*width)
        y2 = int(float(bbox[3].strip())*width)

        if x1 <= 0:
            x1 = 0
        if y1 <= 0:
            y1 = 0
        if x2 >= width:
            x2 = width
        if y2 >= width:
            y2 = width

        crop_tensor = tensor[:, y1:y2, x1:x2]
        crop_tensor = self.tensor_pool(crop_tensor)

        return crop_tensor
