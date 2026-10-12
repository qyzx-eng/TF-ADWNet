from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pywt
import matplotlib.pyplot as plt
import copy
import math
from einops import rearrange
from torch import Tensor

class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients


        if kernel_size is None:

            self.wavelet = pywt.Wavelet(wv)

            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)

            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)

            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)

        self.swt_coeffs_debug = coeffs
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x

        coeffs = []
        dilation = 1
        for _ in range(depth):

            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)

            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2


        coeffs.append(approx_coeffs)
        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        for i in range(m):
            detail_coeff = detail_coeffs[:, :, i, :]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2

        return approx_coeff




def extract_freq_components(wavelet_embedding, x):

    coeffs = wavelet_embedding(x)
    low_freq = coeffs[..., 0, :]
    high_freq = coeffs[..., 1:, :]

    return low_freq, high_freq


# 升维
def adjust_dimensions(input_tensor, m):

    batch_size, channels, length = input_tensor.shape
    input_tensor = input_tensor.unsqueeze(2)
    input_tensor = input_tensor.expand(batch_size, channels, m + 1, length)
    return input_tensor


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x1, x2=None):

        batch_size = x1.size(0)
        hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x1.device)
        cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x1.device)

        out, _ = self.lstm(x1, (hidden_state, cell_state))
        return out




class AdaptiveConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, groups=1):
        super().__init__()

        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c_in, c_out, 1, groups=groups),
            nn.Sigmoid()
        )


        self.base_conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2, groups=groups)


    def forward(self, x):

        x = x.transpose(1, 2)
        w = self.kernel_gen(x)
        out = self.base_conv(x * w)
        out = out.transpose(1, 2)
        return out


class GeomAttentionLayer(nn.Module):
    def __init__(self, configs, attention, d_model,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5, ):
        super(GeomAttentionLayer, self).__init__()

        self.configs = configs
        self.d_channel = d_channel
        self.inner_attention = attention

        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m,
                                    kernel_size=kernel_size)

        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model)
        )


        self.high_freq_proj_q = nn.Linear(configs.m, configs.m)
        self.high_freq_proj_k = nn.Linear(configs.m, configs.m)
        self.high_freq_proj_v = nn.Linear(configs.m, configs.m)

        self.lstm = LSTMModule(input_size=configs.m, hidden_size=configs.m)

        self.conv = AdaptiveConv1d(d_model, d_model, kernel_size=3, groups=d_model)
        self.weights = nn.Parameter(torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32), requires_grad=True)



    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):

        low_freq,high_freq = extract_freq_components(self.swt, queries)


        high_freq = high_freq.permute(0, 1, 3, 2)


        hq = self.high_freq_proj_q(high_freq)
        hk = self.high_freq_proj_k(high_freq)
        hv = self.high_freq_proj_v(high_freq)


        B, C, L, D = hq.shape
        hq = hq.reshape(B * C, L, D)
        hk = hk.reshape(B * C, L, D)
        hv = hv.reshape(B * C, L, D)



        hq = self.lstm(hq)
        hk = self.lstm(hk)
        hv = self.lstm(hv)


        hq = hq.reshape(B, C, L, D)
        hk = hk.reshape(B, C, L, D)
        hv = hv.reshape(B, C, L, D)


        h = (
                self.weights[0] * hq +
                self.weights[1] * hk +
                self.weights[2] * hv
        )

        low_freq_iswt_expand = adjust_dimensions(low_freq, self.configs.m + 1)
        low_freq_iswt = self.swt.swt_reconstruction(low_freq_iswt_expand, self.swt.h0, self.swt.h1, self.swt.m,
                                                               self.swt.kernel_size)

        low_freq_iswt = adjust_dimensions(low_freq_iswt, self.configs.m + 1)
        queries = self.query_projection(low_freq_iswt).permute(0, 3, 2, 1)
        keys = self.key_projection(low_freq_iswt).permute(0, 3, 2, 1)
        values = self.value_projection(low_freq_iswt).permute(0, 3, 2, 1)

        low_freq_iswt, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out_iswt = self.out_projection(low_freq_iswt.permute(0, 3, 2, 1))
        out_iswt = out_iswt.permute(0, 1, 3, 2)
        combined_out = torch.cat([out_iswt, h], dim=-1)
        combined_out = combined_out.mean(dim=-1)
        combined_out = self.conv(combined_out)

        return combined_out, attn



class GeomAttention(nn.Module):
    def __init__(self, configs, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False,
                 alpha=1., ):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.alpha = alpha

    def forward(self, queries, keys, values, attn_mask=None):

        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)


        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries ** 2, dim=-1)
        keys_norm2 = torch.sum(keys ** 2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale



        if self.mask_flag:
            if attn_mask is None:

                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)

            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))


        A = self.dropout(torch.softmax(scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return (V.contiguous(), scores.abs().mean())
