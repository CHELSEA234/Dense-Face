import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from .model_irse import Backbone

import open_clip
from ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    this function is ported from textual inversion official github:
        https://github.com/rinongal/textual_inversion/blob/main/ldm/modules/encoders/modules.py
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _build_causal_attention_mask(bsz, seq_len, dtype):
    '''
            this function is ported from textual inversion official github:
        https://github.com/rinongal/textual_inversion/blob/main/ldm/modules/encoders/modules.py
    '''
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenCLIPEmbedder(AbstractEncoder):
    """
        Uses the CLIP transformer encoder for text (from Hugging Face)
        the core function for the conditional model.
        based on the:
            https://github.com/rinongal/textual_inversion/blob/main/ldm/modules/encoders/modules.py#L157
    """
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

        def embedding_forward(
                self,
                input_ids = None,
                position_ids = None,
                inputs_embeds = None,
                embedding_manager = None,
                reference_img = None,
            ) -> torch.Tensor:

                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]

                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)

                if embedding_manager is not None:
                    # print("the FrozenCLIPEmbedder's embedding_manager: ")
                    inputs_embeds = embedding_manager(reference_img, input_ids, inputs_embeds)

                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                
                return embeddings      

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(self.transformer.text_model.embeddings)

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask = None,
            causal_attention_mask = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(self.transformer.text_model.encoder)


        def text_encoder_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
            reference_img = None
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(
                                            input_ids=input_ids, 
                                            position_ids=position_ids, 
                                            embedding_manager=embedding_manager,
                                            reference_img=reference_img
                                            )

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
            reference_img = None
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager = embedding_manager,
                reference_img = reference_img
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, image, **kwargs):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        z = self.transformer(input_ids=tokens, reference_img=image, **kwargs)
        return z

    def encode(self, text, image, **kwargs):
        '''
            important in the original code when called by:
                hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode)
        '''
        return self(text, image, **kwargs)

class IDLoss(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("/user/guoxia11/cvl/CVPR_personalization/ControlNet/models/model_ir_se50.pth"))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.freeze()

    def freeze(self):
        self.facenet = self.facenet.eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

        self.face_pool = self.face_pool.eval()
        for param in self.face_pool.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        if x.size()[0] == 1:    # GX: bn give error on the single input.
            x = torch.cat([x, x], dim=0)
            split_flag = True
        else:
            split_flag = False
        x = self.face_pool(x)
        x_body_feats, x_feats = self.facenet(x)
        if split_flag:
            x_feats = x_feats[0, :]
            x_feats = torch.unsqueeze(x_feats, 0)
        return x_feats

    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            loss_value  = loss_sample.unsqueeze(0)
            losses.append(loss_sample.unsqueeze(0))
        losses = torch.cat(losses, dim=0)
        return losses

    def encode(self, x, x_hat):
        return self(x, x_hat)
