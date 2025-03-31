"""
MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
该模型结合了市场信息和个股信息进行股票价格预测
论文: https://arxiv.org/abs/2312.15235
"""

import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Tuple, Union, Text
import tqdm
import pprint as pp
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import torch.optim as optim

# 导入PatchTST相关模块
from ..layers.PatchTST_backbone import PatchTST_backbone
from ..layers.PatchTST_layers import series_decomp

import qlib
# from qlib.utils import init_instance_by_config
# from qlib.data.dataset import Dataset, DataHandlerLP, DatasetH
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model as QLibBaseModel


class PatchTSTModel(nn.Module):
    """
    PatchTST模型
    用于处理时间序列数据的Transformer模型
    """
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[torch.Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # 加载参数
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # 模型
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            trend = self.model_trend(trend_init, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    用于为输入序列添加位置信息，使模型能够区分不同位置的特征
    
    参数:
        d_model: 特征维度
        max_len: 最大序列长度
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引向量 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 使用sin和cos函数生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        # 注册为模型的buffer（不作为可训练参数）
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, d_model]
            
        输出:
            x + pe: 原始输入加上位置编码
        """
        return x + self.pe[:x.shape[1], :]


class Gate(nn.Module):
    """
    门控机制 (Feature Gate)
    用于根据市场信息动态调整个股特征的重要性
    
    参数:
        d_input: 输入维度（市场特征维度）
        d_output: 输出维度（个股特征维度）
        beta: 温度参数，控制softmax的平滑度
    """
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        # 线性变换层，将市场特征映射到个股特征空间
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta  # 温度参数

    def forward(self, gate_input):
        """
        输入:
            gate_input: 市场特征 [batch_size, d_input]
            
        输出:
            特征权重 [batch_size, d_output]
        """
        # 线性变换
        output = self.trans(gate_input)
        # 使用softmax将输出转换为权重
        output = torch.softmax(output/self.t, dim=-1)
        # 乘以d_output使权重和保持不变
        return self.d_output*output

class MultiScaleDownsampler(nn.Module):
    """
    多尺度下采样模块
    使用CNN金字塔式下采样，从原始长序列生成多个时间尺度的特征
    从原始序列(32)生成多个时间尺度(32/16/8)
    
    参数:
        d_model: 特征维度
    """
    def __init__(self, d_model):
        super().__init__()
        # 中尺度池化(32 -> 16)
        self.pool_mid = nn.AvgPool1d(kernel_size=2, stride=2)
        # 小尺度池化(32 -> 8)
        self.pool_small = nn.AvgPool1d(kernel_size=4, stride=4)
        
        # 特征转换层，确保不同尺度的特征具有相同的语义
        self.transform_mid = nn.Linear(d_model, d_model)
        self.transform_small = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len(32), d_model]
            
        输出:
            x_large: [batch_size, 32, d_model] - 原始尺度
            x_mid: [batch_size, 16, d_model] - 中等尺度
            x_small: [batch_size, 8, d_model] - 小尺度
        """
        # 调整维度顺序以适应1D池化 [B, L, D] -> [B, D, L]
        x_t = x.transpose(1, 2)
        
        # 应用池化操作
        x_mid_t = self.pool_mid(x_t)  # [B, D, 16]
        x_small_t = self.pool_small(x_t)  # [B, D, 8]
        
        # 恢复原始维度顺序 [B, D, L] -> [B, L, D]
        x_mid = x_mid_t.transpose(1, 2)  # [B, 16, D]
        x_small = x_small_t.transpose(1, 2)  # [B, 8, D]
        
        # 应用特征转换
        x_mid = self.transform_mid(x_mid)
        x_small = self.transform_small(x_small)
        
        return x, x_mid, x_small

class TAttention(nn.Module):
    """
    股票内部注意力模块 (Intra-Stock Attention)
    用于捕捉单个股票内部不同时间步之间的关系，实现股票内部信息的聚合
    使用PatchTST架构处理时间序列数据
    
    参数:
        d_model: 特征维度
        nhead: 多头注意力的头数
        dropout: dropout比率
        patch_len: patch长度
        stride: patch步长
    """
    def __init__(self, d_model, nhead, dropout, patch_len=8, stride=4, seq_len=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 创建配置对象
        class Config:
            def __init__(self):
                self.enc_in = d_model
                self.seq_len = seq_len  # 时间窗口长度，可变
                self.pred_len = 1
                self.e_layers = 1
                self.n_heads = nhead
                self.d_model = d_model
                self.d_ff = 0
                self.dropout = dropout
                self.fc_dropout = dropout
                self.head_dropout = 0
                self.individual = False
                self.patch_len = patch_len
                self.stride = stride
                self.padding_patch = 'end'
                self.revin = 1
                self.affine = 0
                self.subtract_last = False
                self.decomposition = False
                self.kernel_size = 25
        
        configs = Config()
        
        # 创建PatchTST模型
        self.patchtst = PatchTSTModel(
            configs=configs,
            max_seq_len=seq_len,  # 使用对应的序列长度
            d_k=d_model,
            d_v=d_model,
            norm='BatchNorm',
            attn_dropout=dropout,
            act="gelu",
            key_padding_mask='auto',
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe='zeros',
            learn_pe=True,
            pretrain_head=False,
            head_type='flatten',
            verbose=False
        )
        
        # 输入层标准化
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # 前馈网络层标准化
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # 前馈网络
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        输入:
            x: [batch_size, seq_len, d_model]
               其中batch_size表示股票数量，seq_len表示时间步长
            key_padding_mask: [batch_size * d_model, patch_num] 的布尔张量，True表示需要屏蔽的位置
            attn_mask: [seq_len, seq_len] 的布尔张量，限制注意力的访问范围
                
        输出:
            股票内部注意力处理后的特征，捕捉了单个股票不同时间步之间的关系
        """
        # 第一层标准化
        x = self.norm1(x)
        
        # 调整输入维度以适应PatchTST
        # PatchTST期望输入格式为 [batch_size, seq_len, n_vars]
        att_output = x  # 已经是正确的格式 [batch_size, seq_len, d_model]
        
        # 如果提供了patch级别的key_padding_mask
        if key_padding_mask is not None:
            # 注释掉调试信息
            # print(f"[TAttention.forward] 输入x形状: {x.shape}")
            # print(f"[TAttention.forward] 输入key_padding_mask形状: {key_padding_mask.shape}")
            batch_size = x.shape[0]
            
            # 确保key_padding_mask形状与PatchTST内部的[bs*nvars, patch_num]兼容
            # 在PatchTST_backbone中，实际操作是将[bs, nvars, ...] reshape为 [bs*nvars, ...]
            expected_bs_nvars = batch_size * self.d_model
            
            if key_padding_mask.shape[0] != expected_bs_nvars:
                pass
                # print(f"[TAttention.forward] 警告: key_padding_mask形状不匹配 - 预期为[{expected_bs_nvars}, patch_num]")
                # 这里可以选择扩展mask或者不使用mask
        
        # 将掩码信息传递给底层PatchTST模型
        # 注意：这里将key_padding_mask和attn_mask都传递给PatchTST
        att_output = self.patchtst(att_output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # [batch_size, seq_len, d_model]
        
        # 残差连接和前馈网络
        xt = x + att_output  # 残差连接
        xt = self.norm2(xt)  # 第二层标准化
        att_output = xt + self.ffn(xt)  # 前馈网络和残差连接

        return att_output

class SAttention(nn.Module):
    """
    股票间注意力模块 (Inter-Stock Attention)
    用于捕捉不同股票之间的相互关系，实现股票间的信息聚合
    使用多头注意力机制处理股票间的依赖关系
    
    参数:
        d_model: 特征维度
        nhead: 多头注意力的头数
        dropout: dropout比率
    """
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        # 缩放因子，用于调整注意力分数
        self.temperature = math.sqrt(self.d_model/nhead)

        # 线性变换层用于生成query, key, value
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        # 为每个注意力头创建dropout层
        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # 输入层标准化
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # 前馈网络层标准化
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # 前馈网络
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, d_model]
               其中batch_size表示股票数量，seq_len表示时间步长
                
        输出:
            股票间注意力处理后的特征，捕捉了不同股票之间的相互关系
        """
        # 第一层标准化
        x = self.norm1(x)
        # 生成query, key, value并转置
        q = self.qtrans(x).transpose(0,1)  # [seq_len, batch_size, d_model]
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        # 多头注意力机制
        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            # 获取每个头的query, key, value
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # 计算注意力分数并进行softmax
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            # 应用dropout
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            # 计算注意力输出并转置回原始形状
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        # 合并多头注意力输出
        att_output = torch.concat(att_output, dim=-1)

        # 残差连接和前馈网络
        xt = x + att_output  # 残差连接
        xt = self.norm2(xt)  # 第二层标准化
        att_output = xt + self.ffn(xt)  # 前馈网络和残差连接

        return att_output


class TemporalAttention(nn.Module):
    """
    时序注意力模块
    用于聚合不同时间步的特征，生成最终的表示
    
    参数:
        d_model: 特征维度
    """
    def __init__(self, d_model):
        super().__init__()
        # 线性变换层
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        """
        输入:
            z: [batch_size, seq_len, d_model]
            
        输出:
            聚合后的特征 [batch_size, d_model]
        """
        # 线性变换获取表示
        h = self.trans(z)  # [N, T, D]
        # 使用最后一个时间步的特征作为查询向量
        query = h[:, -1, :].unsqueeze(-1)  # [N, D, 1]
        # 计算注意力分数
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] x [N, D, 1] = [N, T]
        # 对注意力分数进行softmax
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # [N, 1, T]
        # 加权聚合所有时间步的特征
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T] x [N, T, D] = [N, 1, D] -> [N, D]
        return output


class MultiScaleFusion(nn.Module):
    """
    多尺度特征融合模块
    用于融合不同时间尺度的特征
    
    参数:
        d_model: 特征维度
    """
    def __init__(self, d_model):
        super().__init__()
        # 特征融合层
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.norm = LayerNorm(d_model, eps=1e-5)
        self.activation = nn.ReLU()

    def forward(self, x_large, x_mid, x_small):
        """
        输入:
            x_large: [batch_size, d_model] - 大尺度特征(32天)
            x_mid: [batch_size, d_model] - 中尺度特征(16天)
            x_small: [batch_size, d_model] - 小尺度特征(8天)
            
        输出:
            融合后的特征 [batch_size, d_model]
        """
        # 拼接三个尺度的特征
        x_concat = torch.cat([x_large, x_mid, x_small], dim=-1)
        # 线性变换进行融合
        x_fused = self.fusion(x_concat)
        # 标准化和激活
        x_fused = self.activation(self.norm(x_fused))
        return x_fused


class MASTER(nn.Module):
    """
    MASTER模型 (Market-Guided Stock Transformer)
    结合市场信息和个股信息的Transformer模型，用于股票价格预测
    
    参数:
        d_feat: 个股特征维度
        d_model: 模型内部特征维度
        t_nhead: 时间注意力头数
        s_nhead: 空间注意力头数
        T_dropout_rate: 时间注意力dropout率
        S_dropout_rate: 空间注意力dropout率
        gate_input_start_index: 市场特征起始索引
        gate_input_end_index: 市场特征结束索引
        beta: 门控机制温度参数
    """
    def __init__(self, d_feat=158, d_model=160, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # 市场特征相关参数
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'市场特征维度
        # 保存d_model参数为实例属性
        self.d_model = d_model
        # 特征门控层
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # 输入映射层
        self.x2y = nn.Linear(d_feat, d_model)
        # 位置编码
        self.pe = PositionalEncoding(d_model, max_len=32)  # 支持32天时间窗口
        
        # 多尺度下采样
        self.downsampler = MultiScaleDownsampler(d_model)
        
        # 单一的时间注意力模块(TAttention)，用于处理所有尺度
        # 配置为最大序列长度32
        # 注意：PatchTST内部的nvars取决于enc_in参数，而enc_in被设置为d_model
        # 这意味着PatchTST会将d_model视为特征维度，导致批次维度变为bs*d_model
        print(f"[__init__] 初始化shared_tattention时设置的enc_in={d_model}，这将成为内部的nvars值")
        self.shared_tattention = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate, 
                                   patch_len=8, stride=4, seq_len=32)
        
        # 空间注意力层
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        
        # 单一的时序注意力模块
        self.temporalatten = TemporalAttention(d_model=d_model)
        
        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFusion(d_model)
        
        # 解码器（输出层）
        self.decoder = nn.Linear(d_model, 1)
        
        # 添加patch掩码缓存，避免重复计算
        self.patch_mask_cache = {}

    def count_parameters(self):
        """
        统计模型各部分的参数量
        """
        total_params = 0
        print("\n模型参数量统计:")
        print("-" * 50)
        
        # 统计特征门控层参数量
        gate_params = sum(p.numel() for p in self.feature_gate.parameters())
        print(f"特征门控层 (Gate): {gate_params:,} 参数")
        total_params += gate_params
        
        # 统计输入映射层参数量
        x2y_params = sum(p.numel() for p in self.x2y.parameters())
        print(f"输入映射层 (x2y): {x2y_params:,} 参数")
        total_params += x2y_params
        
        # 统计下采样层参数量
        downsampler_params = sum(p.numel() for p in self.downsampler.parameters())
        print(f"多尺度下采样层 (MultiScaleDownsampler): {downsampler_params:,} 参数")
        total_params += downsampler_params
        
        # 统计共享时间注意力层参数量
        shared_tatten_params = sum(p.numel() for p in self.shared_tattention.parameters())
        print(f"共享时间注意力层 (SharedTAttention): {shared_tatten_params:,} 参数")
        total_params += shared_tatten_params
        
        # 统计空间注意力层参数量
        satten_params = sum(p.numel() for p in self.satten.parameters())
        print(f"空间注意力层 (SAttention): {satten_params:,} 参数")
        total_params += satten_params
        
        # 统计时序注意力层参数量
        temporalatten_params = sum(p.numel() for p in self.temporalatten.parameters())
        print(f"时序注意力层 (TemporalAttention): {temporalatten_params:,} 参数")
        total_params += temporalatten_params
        
        # 统计特征融合层参数量
        fusion_params = sum(p.numel() for p in self.multi_scale_fusion.parameters())
        print(f"多尺度特征融合层 (MultiScaleFusion): {fusion_params:,} 参数")
        total_params += fusion_params
        
        # 统计解码器参数量
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"解码器 (Decoder): {decoder_params:,} 参数")
        total_params += decoder_params
        
        print("-" * 50)
        print(f"总参数量: {total_params:,}")
        return total_params

    def pad_to_max_len(self, x, valid_len, max_len=32):
        """
        将序列填充到指定的最大长度并创建对应的掩码
        
        参数:
            x: 输入序列 [batch_size, valid_len, d_model]
            valid_len: 序列的有效长度
            max_len: 填充到的最大长度
            
        返回:
            padded: 填充后的序列 [batch_size, max_len, d_model]
            mask: 填充掩码 [batch_size, max_len]，True表示要屏蔽的位置
        """
        batch_size, _, d_model = x.shape
        
        # 如果已经是最大长度，不需要填充
        if valid_len >= max_len:
            return x[:, :max_len, :], None
        
        # 创建填充后的张量
        padded = torch.zeros(batch_size, max_len, d_model, device=x.device)
        padded[:, :valid_len, :] = x  # 拷贝有效部分
        
        # 创建填充掩码，True表示要屏蔽的位置
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
        mask[:, valid_len:] = True  # valid_len之后的位置设为True（需要屏蔽）
        
        return padded, mask
    
    def create_attention_mask(self, seq_len, valid_len, device):
        """
        创建注意力掩码，限制注意力的访问范围
        
        参数:
            seq_len: 序列的总长度
            valid_len: 有效的序列长度
            device: 设备
            
        返回:
            attn_mask: [seq_len, seq_len] 的布尔张量，True表示要屏蔽的位置
        """
        # 使用矩阵操作创建掩码，避免循环
        # 创建列索引矩阵 [seq_len, seq_len]
        col_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq_len, seq_len)
        
        # 创建掩码：列索引 >= valid_len 的位置为 True（需要屏蔽）
        attn_mask = col_indices >= valid_len
        
        return attn_mask

    def analyze_patches(self, real_seq_len, context_window, patch_len, stride, padding_patch='end'):
        """
        分析PatchTST中的patch覆盖情况，确定哪些patch包含真实数据，哪些包含填充数据
        
        参数:
            real_seq_len: 实际序列长度(如16)
            context_window: 模型接收的序列窗口长度(如32)
            patch_len: patch长度(如8)
            stride: patch步长(如4)
            padding_patch: 是否在末尾添加padding patch
            
        返回:
            patch_info: 每个patch的覆盖范围和状态信息的列表
            mask_indices: 建议屏蔽的patch索引列表
        """
        # 1. 计算patch总数
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            patch_num += 1
        
        # 2. 分析每个patch
        patch_info = []
        valid_patches = 0
        partial_patches = 0
        invalid_patches = 0
        
        for i in range(patch_num):
            start_idx = i * stride
            end_idx = start_idx + patch_len - 1
            
            # 计算这个patch中有多少真实数据
            real_data_count = 0
            for j in range(start_idx, end_idx + 1):
                if j < real_seq_len:
                    real_data_count += 1
            
            # 确定patch状态
            if real_data_count == patch_len:
                status = "完全有效"
                valid_patches += 1
            elif real_data_count == 0:
                status = "完全无效"
                invalid_patches += 1
            else:
                status = f"部分有效({real_data_count}/{patch_len})"
                partial_patches += 1
                
            patch_info.append({
                "patch_idx": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "real_data_count": real_data_count,
                "status": status
            })
        
        # 3. 确定哪些patch应该被屏蔽
        # 策略: 真实数据少于一半的patch应该被屏蔽
        mask_indices = [i for i, patch in enumerate(patch_info) if patch["real_data_count"] < patch_len / 2]
        
        # 4. 注释掉打印详细信息
        # print(f"[analyze_patches] 参数: real_seq_len={real_seq_len}, context_window={context_window}, patch_len={patch_len}, stride={stride}")
        # print(f"[analyze_patches] 共{patch_num}个patch: 完全有效={valid_patches}, 部分有效={partial_patches}, 完全无效={invalid_patches}")
        # print(f"[analyze_patches] 建议屏蔽的patch索引: {mask_indices}")
        
        return patch_info, mask_indices

    def build_patch_mask_for_scale(self, batch_size, nvars, real_seq_len, context_window=32, patch_len=8, stride=4, padding_patch='end', device=None, mask_strategy='half'):
        """
        构造精确的patch级别掩码，基于详细的patch分析
        
        参数:
            batch_size: 批次大小
            nvars: 特征维度
            real_seq_len: 实际有效的序列长度
            context_window: 模型接收的上下文窗口长度
            patch_len, stride, padding_patch: 与PatchTST参数一致
            device: 设备
            mask_strategy: 掩码策略('none'=不屏蔽, 'strict'=只要有填充就屏蔽, 'half'=少于一半真实数据就屏蔽)
            
        返回:
            patch_mask: [batch_size * nvars, patch_num]的布尔张量，True表示要屏蔽的patch
        """
        # 创建缓存键
        cache_key = (real_seq_len, context_window, patch_len, stride, padding_patch, mask_strategy)
        
        # 检查缓存中是否已有对应的mask模板
        if cache_key in self.patch_mask_cache:
            # 使用缓存中的mask模板，并根据当前batch_size和nvars进行调整
            mask_template = self.patch_mask_cache[cache_key]
            
            # 扩展模板以适应当前batch_size和nvars
            if nvars > 1:
                # 复制模板到正确的大小 [batch_size*nvars, patch_num]
                patch_mask = mask_template.repeat(batch_size * nvars, 1).to(device)
            else:
                # 仅复制到batch_size大小 [batch_size, patch_num]
                patch_mask = mask_template.repeat(batch_size, 1).to(device)
                
            return patch_mask
        
        # 如果缓存中没有，需要计算新的mask
        # 1) 分析patches
        patch_info, mask_indices = self.analyze_patches(
            real_seq_len, context_window, patch_len, stride, padding_patch
        )
        
        # 2) 计算patch_num
        patch_num = len(patch_info)
        
        # 3) 构造单个mask模板，初始全False(=不屏蔽)
        mask_template = torch.zeros(1, patch_num, dtype=torch.bool)
        
        # 4) 根据分析结果设置掩码
        if mask_indices and mask_strategy != 'none':
            mask_template[:, mask_indices] = True
        
        # 5) 将模板存入缓存
        self.patch_mask_cache[cache_key] = mask_template
        
        # 6) 根据当前batch_size和nvars扩展模板
        # 在PatchTST内部，会将[bs, nvars, patch_num, d_model]重塑为[bs*nvars, patch_num, d_model]
        # 因此掩码需要从[batch_size, patch_num]扩展到[batch_size*nvars, patch_num]
        if nvars > 1:
            # 复制模板到正确的大小 [batch_size*nvars, patch_num]
            patch_mask = mask_template.repeat(batch_size * nvars, 1).to(device)
        else:
            # 仅复制到batch_size大小 [batch_size, patch_num]
            patch_mask = mask_template.repeat(batch_size, 1).to(device)
        
        return patch_mask

    def ensure_patch_mask_shape(self, mask, batch_size, nvars, patch_num, device):
        """
        确保patch掩码形状正确，适配PatchTST内部的[bs*nvars, patch_num]格式
        
        参数:
            mask: 输入掩码，形状为[batch_size, patch_num]
            batch_size: 批次大小
            nvars: 特征维度(=d_model)
            patch_num: patch数量
            device: 设备
            
        返回:
            形状为[batch_size*nvars, patch_num]的掩码
        """
        # 检查输入掩码是否已经是正确形状
        if mask is None:
            return None
            
        # 如果掩码已经是正确形状，直接返回
        if mask.shape[0] == batch_size * nvars:
            return mask
            
        # 如果形状不匹配，需要扩展掩码
        # print(f"[ensure_patch_mask_shape] 扩展掩码: {mask.shape} -> [{batch_size*nvars}, {patch_num}]")
        
        # 使用repeat_interleave复制每行nvars次
        expanded_mask = mask.repeat_interleave(nvars, dim=0)
        
        return expanded_mask

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, feature_dim]
               其中feature_dim包括个股特征和市场特征
               
        输出:
            预测值 [batch_size]
        """
        # 分离个股特征和市场特征
        src = x[:, :, :self.gate_input_start_index]  # 个股特征 [N, T, D]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # 最后时间步的市场特征 [N, D']
        
        # 使用市场特征门控个股特征
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)  # [N, T, D] * [N, 1, D]

        # 特征映射
        x = self.x2y(src)  # [N, T, d_model]
        # 添加位置编码
        x = self.pe(x)
        
        # 多尺度下采样
        x_large, x_mid, x_small = self.downsampler(x)  # [N, 32, d], [N, 16, d], [N, 8, d]
        
        # 获取批次大小和设备
        batch_size = x.shape[0]
        device = x.device
        
        # 在MASTER模型中，nvars是d_model，因为TAttention.patchtst中enc_in被设置为d_model
        # 这意味着PatchTST会将每个特征维度视为独立通道
        nvars = self.d_model
        
        # 注释掉调试信息打印
        # print(f"[forward] batch_size={batch_size}, d_model={self.d_model}, nvars={nvars}")
        
        # 填充中尺度和小尺度序列到最大长度(32)，并创建原始key_padding_mask
        x_mid_padded, _ = self.pad_to_max_len(x_mid, valid_len=16)
        x_small_padded, _ = self.pad_to_max_len(x_small, valid_len=8)
        
        # 构建精确的patch级别key_padding_mask
        # 对于大尺度(32步)，无需构建掩码
        
        # 对于中尺度(16步)，构建掩码以屏蔽不完全包含真实数据的patch
        patch_mask_mid = self.build_patch_mask_for_scale(
            batch_size, 1, real_seq_len=16, context_window=32,
            patch_len=8, stride=4, padding_patch='end', device=device
        )
        
        # 对于小尺度(8步)，构建掩码以屏蔽不完全包含真实数据的patch
        patch_mask_small = self.build_patch_mask_for_scale(
            batch_size, 1, real_seq_len=8, context_window=32,
            patch_len=8, stride=4, padding_patch='end', device=device
        )
        
        # 确保掩码形状正确，以适配PatchTST内部[bs*nvars, patch_num]的期望
        patch_mask_mid = self.ensure_patch_mask_shape(patch_mask_mid, batch_size, nvars, patch_mask_mid.shape[1], device)
        patch_mask_small = self.ensure_patch_mask_shape(patch_mask_small, batch_size, nvars, patch_mask_small.shape[1], device)
        
        # 1. 使用共享的时间注意力模块处理不同尺度的序列
        # 大尺度：无需掩码，可以看到全部32步
        x_large_processed = self.shared_tattention(x_large)
        
        # 中尺度：使用patch级别key_padding_mask限制
        x_mid_processed = self.shared_tattention(
            x_mid_padded, 
            key_padding_mask=patch_mask_mid
        )
        
        # 小尺度：使用patch级别key_padding_mask限制
        x_small_processed = self.shared_tattention(
            x_small_padded, 
            key_padding_mask=patch_mask_small
        )
        
        # 2. 应用空间注意力处理不同股票间的关系
        x_large_spatial = self.satten(x_large_processed)  # [N, 32, d_model]
        x_mid_spatial = self.satten(x_mid_processed)  # [N, 32, d_model]
        x_small_spatial = self.satten(x_small_processed)  # [N, 32, d_model]
        
        # 截取回有效长度
        x_mid_spatial = x_mid_spatial[:, :16, :]
        x_small_spatial = x_small_spatial[:, :8, :]
        
        # 3. 应用时序注意力聚合时间维度
        x_large_agg = self.temporalatten(x_large_spatial)  # [N, d_model]
        x_mid_agg = self.temporalatten(x_mid_spatial)      # [N, d_model]
        x_small_agg = self.temporalatten(x_small_spatial)  # [N, d_model]
        
        # 多尺度特征融合
        x_fused = self.multi_scale_fusion(x_large_agg, x_mid_agg, x_small_agg)  # [N, d_model]
        
        # 解码输出
        output = self.decoder(x_fused).squeeze(-1)  # [N]

        return output

def calc_ic(pred, label):
    """
    计算IC和Rank IC
    
    参数:
        pred: 预测值
        label: 真实标签
        
    返回:
        ic: 信息系数
        ric: 排名信息系数
    """
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    """
    按日期批量采样器
    将同一天的样本分为一批，用于模型训练
    
    参数:
        data_source: 数据源
        shuffle: 是否打乱日期顺序
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # 计算每个批次的样本数（每天的股票数量）
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        # 计算每个批次的起始索引
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0
        
        # 添加调试信息
        print(f"总批次数: {len(self.daily_count)}")
        print(f"每个批次的样本数: {self.daily_count}")
        print(f"每个批次的起始索引: {self.daily_index}")

    def __iter__(self):
        """
        迭代器，返回每一批的索引
        """
        if self.shuffle:
            # 打乱日期顺序
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                batch_indices = np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
                print(f"生成第{i}个批次，包含{len(batch_indices)}个样本")
                yield batch_indices
        else:
            # 按原始日期顺序
            for idx, count in zip(self.daily_index, self.daily_count):
                batch_indices = np.arange(idx, idx + count)
                print(f"生成第{idx}个批次，包含{len(batch_indices)}个样本")
                yield batch_indices

    def __len__(self):
        return len(self.data_source)


class MASTERModel(QLibBaseModel):
    """
    MASTER模型的训练和预测包装类
    
    参数:
        d_feat: 个股特征维度
        d_model: 模型内部特征维度
        t_nhead: 时间注意力头数
        s_nhead: 空间注意力头数
        gate_input_start_index: 市场特征起始索引
        gate_input_end_index: 市场特征结束索引
        T_dropout_rate: 时间注意力dropout率
        S_dropout_rate: 空间注意力dropout率
        beta: 门控机制温度参数
        n_epochs: 训练轮数
        lr: 学习率
        GPU: GPU设备ID
        seed: 随机种子
        train_stop_loss_thred: 训练停止阈值
        save_path: 模型保存路径
        save_prefix: 模型保存前缀
        benchmark: 基准指数
        market: 市场
        only_backtest: 是否只进行回测
        grad_accum_steps: 梯度累积步数，用于减少内存占用
    """
    def __init__(self, d_feat: int = 158, d_model: int = 160, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=158, gate_input_end_index=221,
            T_dropout_rate=0.5, S_dropout_rate=0.5, beta=None, n_epochs = 40, lr = 8e-6, GPU=0, seed=0, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', benchmark = 'SH000300', market = 'csi300', only_backtest = False, grad_accum_steps = 1):
        
        super().__init__()
        
        # 模型结构参数
        self.d_model = d_model  # 使用256以保持较高的模型表达能力
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        # 训练参数
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.grad_accum_steps = grad_accum_steps  # 梯度累积步数
        
        # 市场相关参数
        self.benchmark = benchmark
        self.market = market
        self.infer_exp_name = f"{self.market}_MASTER_MultiScale_seed{self.seed}_backtest"

        # 模型状态
        self.fitted = False
        
        # 根据市场设置beta值
        if self.market == 'csi300':
            self.beta = 10
        else:
            self.beta = 5
            
        # 设置随机种子
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        # 初始化MASTER模型
        self._model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        # 打印模型参数量
        self._model.count_parameters()
        # 初始化优化器
        self.train_optimizer = optim.Adam(self._model.parameters(), self.lr)
        # 将模型移至指定设备
        self._model.to(self.device)

        # 模型保存相关参数
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.only_backtest = only_backtest

    @property
    def model(self):
        return self._model

    def init_model(self):
        """
        初始化模型和优化器
        """
        if self._model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self._model.parameters(), self.lr)
        self._model.to(self.device)

    def load_model(self, param_path):
        """
        加载预训练模型参数
        
        参数:
            param_path: 模型参数文件路径
        """
        try:
            self._model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.") 

    def loss_fn(self, pred, label):
        """
        损失函数（均方误差）
        
        参数:
            pred: 预测值
            label: 真实标签
            
        返回:
            均方误差
        """
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        """
        初始化数据加载器
        
        参数:
            data: 数据集
            shuffle: 是否打乱
            drop_last: 是否丢弃最后一个不完整批次
            
        返回:
            数据加载器
        """
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # 添加batch_size参数
        data_loader = DataLoader(data, batch_size=1, sampler=sampler, drop_last=drop_last)
        return data_loader

    def train_epoch(self, data_loader):
        """
        训练一个epoch
        
        参数:
            data_loader: 数据加载器
            
        返回:
            平均训练损失
        """
        self._model.train()
        losses = []

        # 添加调试信息
        print("开始训练一个epoch...")
        
        # 梯度累积相关参数
        batch_count = 0
        accumulated_loss = 0
        
        for i, data in enumerate(data_loader):
            if i == 0:
                print(f"第一个batch的shape: {data.shape}")
            
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - 股票数量
            T - 回溯窗口长度，32（用于多尺度时间建模：32/16/8）
            F - 158个因子 + 63个市场信息 + 1个标签          
            '''
            # 提取特征和标签
            feature = data[:, :, 0:-1].to(self.device)  # 特征 [N, T, F-1]
            label = data[:, -1, -1].to(self.device)     # 标签 [N]
            
            if torch.any(torch.isnan(label)):
                print(f"警告：第{i}个batch包含NaN标签")
                continue

            # 打印特征形状以确认时间窗口长度
            if i == 0:
                print(f"输入特征的形状: {feature.shape}，其中时间窗口长度为 {feature.shape[1]}")

            pred = self._model(feature.float())
            loss = self.loss_fn(pred, label)
            # 对损失进行缩放以适应梯度累积
            loss = loss / self.grad_accum_steps
            losses.append(loss.item() * self.grad_accum_steps)  # 保存原始损失值
            
            # 反向传播
            loss.backward()
            
            # 梯度累积逻辑
            batch_count += 1
            accumulated_loss += loss.item()
            
            # 累积到指定步数或最后一个batch时更新参数
            if batch_count % self.grad_accum_steps == 0 or (i + 1) == len(data_loader):
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
                # 更新参数
                self.train_optimizer.step()
                # 清零梯度
                self.train_optimizer.zero_grad()
                
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1} 个batch，当前平均loss: {accumulated_loss/batch_count:.6f}")
                
                # 重置累积损失
                accumulated_loss = 0
                batch_count = 0

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        """
        测试一个epoch
        
        参数:
            data_loader: 数据加载器
            
        返回:
            平均测试损失
        """
        self._model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            # 提取特征和标签
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            # 前向传播
            pred = self._model(feature.float())
            # 计算损失
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def load_param(self, param_path):
        """
        加载模型参数
        
        参数:
            param_path: 参数文件路径
        """
        self._model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dataset: DatasetH, progress_callback=None):
        """
        训练模型
        
        参数:
            dataset: 数据集
            progress_callback: 进度回调函数，用于显示训练进度
        """
        # 准备训练集和验证集
        print("准备训练集和验证集...")
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        print(f"训练集大小: {len(dl_train)}")
        print(f"验证集大小: {len(dl_valid)}")
        
        # 创建数据加载器
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        best_val_loss = 1e3

        # 训练循环
        for step in range(self.n_epochs):
            print(f"\n开始训练第 {step+1}/{self.n_epochs} 个epoch...")
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            # 验证一个epoch
            val_loss = self.test_epoch(valid_loader)

            # 更新进度显示
            if progress_callback:
                progress_callback(step, train_loss, val_loss)
            else:
                print(f"Epoch {step}, train_loss {train_loss:.6f}, valid_loss {val_loss:.6f}")

            # 保存最佳模型
            if best_val_loss > val_loss:
                best_param = copy.deepcopy(self._model.state_dict())
                best_val_loss = val_loss
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")

            # 达到训练停止阈值则提前结束
            if train_loss <= self.train_stop_loss_thred:
                print(f"达到训练停止阈值，提前结束训练")
                break

        # 保存最佳模型
        torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')
        print(f"训练完成，模型已保存到 {self.save_path}{self.save_prefix}master_{self.seed}.pkl")

    def predict(self, dataset: DatasetH, use_pretrained = True):
        """
        模型预测
        
        参数:
            dataset: 数据集
            use_pretrained: 是否使用预训练模型
            
        返回:
            预测结果DataFrame
        """
        # 加载预训练模型
        if use_pretrained:
            self.load_param(f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        # 准备测试集
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        pred_all = []

        # 预测
        self._model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self._model(feature.float()).detach().cpu().numpy()
            pred_all.append(pred.ravel())

        # 将预测结果转换为DataFrame
        pred_all = pd.DataFrame(np.concatenate(pred_all), index=dl_test.get_index())
        # pred_all = pred_all.loc[self.label_all.index]
        # rec = self.backtest()
        return pred_all
