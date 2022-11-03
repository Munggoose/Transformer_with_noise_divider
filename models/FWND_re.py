import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
# from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.NosieDivider import NosieDivider
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FWND(nn.Module):
    

    def __init__(self,configs):
        super(FWND, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        #Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size,list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        
        ##Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, 
                                                configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in,configs.d_model, configs.embed,
                                                configs.freq, configs.dropout)
        
        self.noise_divider = NosieDivider(None)
        
        #Fourior
        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
        decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len//2+self.pred_len,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                    out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len//2+self.pred_len,
                                                    seq_len_kv=self.seq_len,
                                                    modes=configs.modes,
                                                    mode_select_method=configs.mode_select)
        
        noise_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
        
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )


        self.noise_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        noise_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )


        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
    
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = trend_init.transpose(1,2)
        trend_init, noise_init ,trend_mask, noise_mask  = self.noise_divider(trend_init)
        trend_init = trend_init.transpose(1,2)
        noise_init = noise_init.transpose(1,2)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))


        # enc
        enc_out = x_enc - noise_init
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_noise =self.enc_embedding(noise_init, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_noise, noise_attns = self.noise_encoder(enc_noise, attn_mask=enc_self_mask)
        enc_out = enc_out + enc_noise
    
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, noise_attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]