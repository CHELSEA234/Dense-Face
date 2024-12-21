import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


## helper function:
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            reference_delta=0.01,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()
        self.initial_embeddings = nn.ParameterDict() # These should not be optimized, record the original embedding.

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token
        self.is_clip = True
        get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
        get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
        token_dim = 768

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)
            init_word_token = get_token_for_string(initializer_words[idx])

            with torch.no_grad():
                init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

            token_params = torch.nn.Parameter(
                                            init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), 
                                            requires_grad=False
                                            )
            self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                                                                        init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), 
                                                                        requires_grad=False
                                                                        )

            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

        ## convert reference image to the text embedding.
        self.reference_delta = reference_delta
        ## final 512 projector.
        self.projector = nn.Linear(512, token_dim, bias=False)
        self.projector_norm = nn.BatchNorm1d(token_dim)
        # self.output_layer = Sequential(BatchNorm2d(512),
        #                                Dropout(0.6),
        #                                Flatten(),
        #                                Linear(512 * 7 * 7, token_dim),
        #                                BatchNorm1d(token_dim, affine=True))

    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def forward(self, reference_img, tokenized_text, embedded_text):
        '''
            this function is taken place by the new one with the facenet inside.
        '''
        b, n, device = *tokenized_text.shape, tokenized_text.device
        # print("embedding forward function.")
        # for placeholder_string, placeholder_token in self.string_to_token_dict.items():
        #     print(placeholder_string, placeholder_token)
        # print("the string: ")
        # print("token: ", tokenized_text)    # a string contains multiple numbers.
        # print("embed text: ", embedded_text.size()) # embed text:  torch.Size([10, 77, 768])
        # import sys;sys.exit(0)
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)    
            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            embedded_text[placeholder_idx] = placeholder_embedding
        return embedded_text

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
                
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss