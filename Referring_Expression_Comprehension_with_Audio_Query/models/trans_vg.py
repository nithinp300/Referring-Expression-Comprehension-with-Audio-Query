import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .audio_model.wavlm import build_wavlm
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args) #ok
        # self.textmodel = build_bert(args)
        self.audiomodel = build_wavlm(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        # self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.audio_proj = nn.Linear(self.audiomodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, img_data, audio_data):
        # print("img_data", img_data)
        # print("img_data.shape", img_data.tensors.shape)
        bs = img_data.tensors.shape[0]

        # print("bs", bs)
        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        # print("visu_mask", visu_mask.shape, visu_mask)
        # print("visu_src 1 (N*B)xC", visu_src.shape, visu_src) 
        visu_src = self.visu_proj(visu_src) # (N*B)xC
        # print("visu_src 2 (N*B)xC", visu_src.shape, visu_src)
        
        # we need audio_data
        # print("audio_data", audio_data.shape, audio_data)
        audio_src = self.audiomodel(audio_data).last_hidden_state
        # print("audio_src", audio_src.shape, audio_src)
        audio_src = self.audio_proj(audio_src)
        # print("audio_src after proj", audio_src.shape, audio_src)
        audio_src = audio_src.permute(1, 0, 2)
        # print("audio_src after permute", audio_src.shape, audio_src)
        audio_mask = torch.zeros((bs, audio_src.shape[0])).to(audio_src.device).to(torch.bool)
        # print("audio_mask", audio_mask.shape, audio_mask)
        # language bert
        # print("text_data", text_data.shape, text_data)
        # text_fea = self.textmodel(text_data)
        # print("text_fea", text_fea.shape, text_fea)
        # text_src, text_mask = text_fea.decompose()
        # print("text_src", text_src.shape, text_src)
        # print("text_mask", text_mask.shape, text_mask)
        # assert text_mask is not None
        # text_src = self.text_proj(text_src)
        # print("text_src after t_proj", text_src.shape, text_src)
        # # permute BxLenxC to LenxBxC
        # text_src = text_src.permute(1, 0, 2)
        # print("text_src after permute", text_src.shape, text_src)
        # text_mask = text_mask.flatten(1)
        # print("text_mask flatten", text_mask.shape, text_mask)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        # print("tgt_src", tgt_src.shape, tgt_src)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        # print("tgt_mask", tgt_mask.shape, tgt_mask)

        vl_src = torch.cat([tgt_src, audio_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, audio_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
