import argparse, os, sys
import numpy as np
import torch
import pytorch_lightning as pl
import einops
import cv2

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tutorial_dataset_tiny import MyDatasetFace ## open-source demo
from functools import partial
from glob import glob
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sixdrepnet import SixDRepNet
from ldm.util import instantiate_from_config
from src.logger import ImageLogger, ModeSwapCallback

def load_model_from_config(config, ckpt, verbose=False, model=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    try:
        sd = pl_sd["state_dict"]    # loading version 1.5
    except:     
        sd = pl_sd      # loading controlNet stable diffusion.
    config.model.params.ckpt_path = ckpt
    if model is None:
        model = instantiate_from_config(config.model)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Please ignore the content above.")
    m, u = model.load_state_dict(sd, strict=False)
    print(f"{len(m)} keys missing and {len(u)} keys unexpected.")
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument('--resume_path', type=str, 
                        default='./ckpt/epoch=44397232.ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus_num', type=int, default=2)
    parser.add_argument('--logger_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_steps', type=lambda x: int(float(x)), default=6100)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('-elr', '--embedding_learning_rate', type=float, default=5e-3,
                        help='learning rate to train the embedding manager.')
    parser.add_argument('--delta', type=float, default=1e-3, 
                        help='detla on the feature perturbation.')
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(),)
    parser.add_argument("--placeholder_string", type=str, 
                        help="Placeholder string which will be used to denote the concept in future prompts. \
                                Overwrites the config options.")
    parser.add_argument("--init_word", type=str, 
                        help="Word to use as source for initial token embedding")
    parser.add_argument("--embedding_manager_ckpt", type=str, default="", 
                        help="Initialize embedding manager from a checkpoint")
    parser.add_argument('--opt_embed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--keep_prompt', action='store_true')
    parser.add_argument('--sample_num', type=int, default=100)

    ## inference arguments: 
    parser.add_argument('--partition', type=int, default=0, choices=[0,1,2,3,4])
    parser.add_argument('--dir_name', type=str, default='DenseFace_ckpt')
    parser.add_argument('--dump_folder', type=str, default='output')
    parser.add_argument('--subject_dir', type=str, default="./reference_id/")
    parser.add_argument('--face_dir', type=str, default="./cropped_face/")
    parser.add_argument('--mask_dir', type=str, default="./mask/")
    return parser

if __name__ == "__main__":

    # init and save configs
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed_everything(23)
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # setup the directory name
    embedding_only = False
    des_str = '_lr_' + str(opt.learning_rate) \
            + '_embedlr_' + str(opt.embedding_learning_rate) \
            + '_detla_' + str(opt.delta) \
            + '_optembed_' + str(opt.opt_embed) \
            + '_keepPrompt_' + str(opt.keep_prompt)

    # the main model and embedding_manager_ckpt.
    # Note, please set up the personalization_config first before the model initialization.
    config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
    config.model.params.personalization_config.params.placeholder_strings = [opt.placeholder_string]
    config.model.params.personalization_config.params.initializer_words[0] = opt.init_word
    config.model.params.personalization_config.params.reference_delta = opt.delta
    model = load_model_from_config(config, opt.resume_path) # loading the pre-trained embedding.

    # configure learning rate
    scale_value = opt.accumulate_grad_batches * opt.gpus_num * opt.batch_size
    if not embedding_only:
        print("Not embedding only mode.")
        model.learning_rate = scale_value * opt.learning_rate
        model.embedding_learning_rate = scale_value * opt.embedding_learning_rate
    else:
        print("Embedding only mode.")
        model.learning_rate = scale_value * opt.embedding_learning_rate
        raise ValueError   
    model.opt_embed = opt.opt_embed
    model.feature_delta = opt.delta

    ## setup model.
    model._adding_per_kv()
    model.eval()
    model.cuda()
    
    # data; a placeholder to the blending process and obtaining the latent space feature.
    dataset = MyDatasetFace(dataset='CASIA',
                            data_dir="../CASIA_tiny/",
                            keep_prompt=False,
                            csv_name=f"casiaFacesDataset256_{opt.sample_num}_samples_headpose.csv",
                            interval=10e6,
                            dict_file_name=f'CASIA_subject_{opt.sample_num}_samples.txt',
                            mode='eval') 
    dataloader = DataLoader(dataset, 
                            num_workers=opt.num_workers, 
                            batch_size=1, 
                            shuffle=True)

    ## pre-defined hyperparameters:
    N=8 
    n_row=4
    ddim_eta = 1.
    use_ddim = True
    ddim_steps = 50

    @torch.no_grad()
    def sample_log_new(self, init_image, ref_img_feature, mask, orig_mask, cond, batch_size, ddim, ddim_steps, **kwargs):
        '''
            entering the new blending sampling here.
        '''
        from ldm.models.diffusion.ddim_blend import DDIMSampler
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, ref_img_feature=ref_img_feature,
                                                    init_image=init_image, mask=mask, org_mask=orig_mask, 
                                                    verbose=False, **kwargs)
        return samples, intermediates

    model.sample_log = sample_log_new.__get__(model)

    ## obtain the arcface feature from the test samples.
    subject_lst = []
    for sub_idx in range(1,4):
        subject_lst.append(str(sub_idx).zfill(3))
    subject_lst.sort()
    print(subject_lst)

    ## head pose model
    model_headpose = SixDRepNet()

    ## begin inference.
    with torch.no_grad():        
        img_list = glob(opt.face_dir + "*.png")
        img_list.sort()
        for img_idx, img_path in enumerate(img_list):
            mask_path = img_path.replace(opt.face_dir, opt.mask_dir)
            img_name = img_path.split('/')[-1].split('.')[0]
            prompt_id = img_name.split("_")[1]
            prompt_sub_id = img_name.split("_")[2]
            print("img_name: ", img_name, " prompt_ids: ", prompt_id, prompt_sub_id, opt.partition)

            target = cv2.imread(img_path)
            target = cv2.resize(target, (256,256), interpolation = cv2.INTER_AREA)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = (target.astype(np.float32) / 127.5) - 1.0
            pseduo_mask_RGB = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pseduo_mask_RGB = cv2.resize(pseduo_mask_RGB, (32,32), interpolation = cv2.INTER_AREA)
            pseduo_mask_RGB = pseduo_mask_RGB.astype(np.float32) / 255.0

            pitch, yaw, roll = model_headpose.predict(cv2.imread(img_path))
            source = dataset.draw_axis(yaw[0], pitch[0], roll[0])
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = source.astype(np.float32) / 255.0

            target_tensor = torch.from_numpy(target)
            source_tensor = torch.from_numpy(source)
            p_mask_tensor = torch.from_numpy(pseduo_mask_RGB)
            p_mask_tensor = torch.unsqueeze(torch.unsqueeze(p_mask_tensor, -1), 0)

            for b_idx, batch in enumerate(dataloader):
                for subject_num in subject_lst:
                    feat_lst = glob(f"{opt.subject_dir}/{subject_num}/feat_file_v2/*.pth")
                    feat_lst.sort()
                    for feat_idx, feat_file in enumerate(feat_lst):
                        feat_num = feat_file.replace(f"{opt.subject_dir}/{subject_num}/feat_file_v2/", "").replace('.pth', "")
                        target_image_name = f"{opt.dump_folder}/{img_name}_{subject_num}_{feat_num}.png"

                        if os.path.exists(target_image_name):
                            continue

                        batch['refer'] = torch.from_numpy(torch.load(feat_file))
                        batch['jpg'][0] = target_tensor
                        batch['hint'][0] = source_tensor

                        use_ddim = ddim_steps is not None
                        log = dict()
                        batch_tmp = dict()
                        for key, value in batch.items():
                            value = batch[key]
                            if not isinstance(value, list):
                                value = value.cuda()
                            batch_tmp[key] = value
                        batch = batch_tmp

                        target_img_bbox = batch['bbox_t']
                        ref_img_feature = batch['refer_img']
                        z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                                            return_first_stage_outputs=True,
                                                            force_c_encode=True,
                                                            return_original_cond=True,
                                                            bs=N, ref_img_feature=batch['refer'])
                        
                        n_row = min(x.shape[0], n_row)
                        refer_ids = einops.rearrange(ref_img_feature, 'b h w c -> b c h w').contiguous()

                        ## get the control input.
                        N = min(x.shape[0], N)
                        control = batch[model.control_key]
                        if N is not None:
                            control = control[:N]

                        orig_mask = einops.rearrange(p_mask_tensor.cuda(), 'b h w c -> b c h w').contiguous()
                        mask = torch.nn.functional.interpolate(orig_mask, size=(32, 32))
                        mask[mask<0.75] = 0
                        mask[mask>=0.75] = 1.

                        control = control.cuda()
                        control = einops.rearrange(control, 'b h w c -> b c h w')
                        c_cat = control.to(memory_format=torch.contiguous_format).float()

                        # get denoise row
                        unconditional_guidance_scale=4
                        uc = model.get_learned_conditioning(N * [""], ref_img_feature="Dummy")

                        samples, z_denoise_row = model.sample_log(
                                                                init_image=einops.rearrange(batch['jpg'], 'b h w c -> b c h w').contiguous(),  
                                                                mask=mask,  # new
                                                                orig_mask=orig_mask,    # new
                                                                ref_img_feature=batch['refer'][:N], # TODO: ugly code here.
                                                                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                                batch_size=N,ddim=use_ddim,
                                                                ddim_steps=ddim_steps,eta=ddim_eta,
                                                                unconditional_conditioning={"c_concat": [c_cat], "c_crossattn": [uc]},
                                                                unconditional_guidance_scale=unconditional_guidance_scale
                                                                )
                        samples_cfg = model.decode_first_stage(samples)
                        log["samples_scaled"] = samples_cfg
                        os.makedirs(f'{opt.dump_folder}', exist_ok=True)
                        for key, value in log.items():
                            value = (einops.rearrange(value, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                            img = cv2.cvtColor(value[0], cv2.COLOR_BGR2RGB)
                            cv2.imwrite(f"{opt.dump_folder}/{img_name}_{subject_num}_{feat_num}.png", img)