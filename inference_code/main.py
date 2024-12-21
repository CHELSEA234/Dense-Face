import argparse, os, sys
import numpy as np
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf
from torch.utils.data import DataLoader 
# from tutorial_dataset import MyDatasetFace  ## T2I-DenseFace
# from tutorial_dataset_webface import MyDatasetFace  ## web-face
from tutorial_dataset_tiny import MyDatasetFace ## open-source demo
from functools import partial

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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
                        default='./ckpt/DenseFace_ini.ckpt')
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
    config_yaml = opt.base[0].split('/')[-1]
    dir_name = ""
    embedding_only = False
    if opt.debug:
        dir_name = dir_name + "debug"

    des_str = '_lr_' + str(opt.learning_rate) \
            + '_embedlr_' + str(opt.embedding_learning_rate) \
            + '_detla_' + str(opt.delta) \
            + '_sampleNum_ALL' \
            + '_optembed_' + str(opt.opt_embed) \
            + '_steps_' + str(opt.max_steps) \
            + '_keepPrompt_' + str(opt.keep_prompt)
    dir_name = dir_name + des_str

    config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
    config.model.params.personalization_config.params.placeholder_strings = [opt.placeholder_string]
    config.model.params.personalization_config.params.initializer_words[0] = opt.init_word
    config.model.params.personalization_config.params.reference_delta = opt.delta
    
    model = load_model_from_config(config, opt.resume_path) # loading the pre-trained embedding.

    # configure learning rate
    ## configure_optimizers_KV uses self._adding_per_kv() and self.learning_rate
    ## configure_optimizers_embed only updates the perturbation and self.embedding_learning_rate
    scale_value = opt.accumulate_grad_batches * opt.gpus_num * opt.batch_size
    if not embedding_only:
        print("Not embedding only mode.")
        model.learning_rate = scale_value * opt.learning_rate
        model.embedding_learning_rate = scale_value * opt.embedding_learning_rate
    else:
        print("Embedding only mode.")
        model.learning_rate = scale_value * opt.embedding_learning_rate
        raise ValueError    # GX: does it mean this option deprecates?
    model.opt_embed = opt.opt_embed
    model.feature_delta = opt.delta

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

    # Misc
    logger = ImageLogger(
                        batch_frequency=opt.logger_freq, 
                        max_images=4, 
                        increase_log_steps=False
                        )
    swap_model = ModeSwapCallback(swap_step=0, is_frozen=True)  # do configure_optimizers_KV
    ckpt_callback = ModelCheckpoint(
                                    every_n_train_steps=opt.save_freq, 
                                    # monitor='train/loss_id',
                                    # save_top_k=args.save_top_k
                                    save_top_k=-1,
                                    verbose=True
                                    )

    # train
    trainer = pl.Trainer(
                        strategy="ddp", 
                        accelerator="gpu", 
                        devices=opt.gpus_num, 
                        precision=16,
                        # replace_sampler_ddp=False,  # https://github.com/Lightning-AI/lightning/issues/11054#issuecomment-1019515550
                        # max_steps=opt.max_steps,
                        max_epochs=opt.max_steps,
                        callbacks=[logger, ckpt_callback, swap_model],
                        default_root_dir=dir_name
                        ) 
    trainer.fit(model, train_dataloader=dataloader)