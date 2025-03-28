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
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
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
    def __init__(self, d_model, nhead, dropout, patch_len=8, stride=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 创建配置对象
        class Config:
            def __init__(self):
                self.enc_in = d_model
                self.seq_len = 8  # 修改为与输入序列长度匹配
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
            max_seq_len=8,  # 修改为与输入序列长度匹配
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

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, d_model]
               其中batch_size表示股票数量，seq_len表示时间步长
                
        输出:
            股票内部注意力处理后的特征，捕捉了单个股票不同时间步之间的关系
        """
        # 第一层标准化
        x = self.norm1(x)
        
        # 调整输入维度以适应PatchTST
        # PatchTST期望输入格式为 [batch_size, seq_len, n_vars]
        att_output = x  # 已经是正确的格式 [batch_size, seq_len, d_model]
        
        # 应用PatchTST
        att_output = self.patchtst(att_output)  # [batch_size, seq_len, d_model]
        
        # 残差连接和前馈网络
        xt = x + att_output  # 残差连接
        xt = self.norm2(xt)  # 第二层标准化
        att_output = xt + self.ffn(xt)  # 前馈网络和残差连接

        return att_output

class SAttention(nn.Module):
    """
    股票间注意力模块 (Inter-Stock Attention)
    用于捕捉不同股票之间的相互关系，实现股票间的信息聚合
    使用PatchTST架构处理股票间的依赖关系
    
    参数:
        d_model: 特征维度
        nhead: 多头注意力的头数
        dropout: dropout比率
    """
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        
        # 创建配置对象
        class Config:
            def __init__(self):
                self.enc_in = d_model
                self.seq_len = 8  # 假设序列长度为8
                self.pred_len = 1
                self.e_layers = 1
                self.n_heads = nhead
                self.d_model = d_model
                self.d_ff = 0
                self.dropout = dropout
                self.fc_dropout = dropout
                self.head_dropout = 0
                self.individual = False
                self.patch_len = 4  # 可根据需要调整
                self.stride = 2    # 可根据需要调整
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
            max_seq_len=8,
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
        
        # 调整输入维度以适应PatchTST
        # 对于股票间注意力，需要转置输入使股票作为序列处理
        x_trans = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # 转回PatchTST期望的输入格式 [batch_size, seq_len, d_model]
        att_input = x_trans.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 应用PatchTST
        att_output = self.patchtst(att_input)  # [batch_size, seq_len, d_model]
        
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
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # 市场特征相关参数
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'市场特征维度
        # 特征门控层
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # 输入映射层
        self.x2y = nn.Linear(d_feat, d_model)
        # 位置编码
        self.pe = PositionalEncoding(d_model)
        # 时间注意力层
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        # 空间注意力层
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        # 时序注意力层（用于聚合时间步）
        self.temporalatten = TemporalAttention(d_model=d_model)
        # 解码器（输出层）
        self.decoder = nn.Linear(d_model, 1)

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
        
        # 统计时间注意力层参数量
        tatten_params = sum(p.numel() for p in self.tatten.parameters())
        print(f"时间注意力层 (TAttention): {tatten_params:,} 参数")
        total_params += tatten_params
        
        # 统计空间注意力层参数量
        satten_params = sum(p.numel() for p in self.satten.parameters())
        print(f"空间注意力层 (SAttention): {satten_params:,} 参数")
        total_params += satten_params
        
        # 统计时序注意力层参数量
        temporalatten_params = sum(p.numel() for p in self.temporalatten.parameters())
        print(f"时序注意力层 (TemporalAttention): {temporalatten_params:,} 参数")
        total_params += temporalatten_params
        
        # 统计解码器参数量
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"解码器 (Decoder): {decoder_params:,} 参数")
        total_params += decoder_params
        
        print("-" * 50)
        print(f"总参数量: {total_params:,}")
        return total_params

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
        # 应用时间注意力
        x = self.tatten(x)
        # 应用空间注意力
        x = self.satten(x)
        # 应用时序注意力聚合时间维度
        x = self.temporalatten(x)  # [N, d_model]
        # 解码输出
        output = self.decoder(x).squeeze(-1)  # [N]

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
    """
    def __init__(self, d_feat: int = 158, d_model: int = 256, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=158, gate_input_end_index=221,
            T_dropout_rate=0.5, S_dropout_rate=0.5, beta=None, n_epochs = 40, lr = 8e-6, GPU=0, seed=0, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', benchmark = 'SH000300', market = 'csi300', only_backtest = False):
        
        super().__init__()
        
        # 模型结构参数
        self.d_model = d_model
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
        
        # 市场相关参数
        self.benchmark = benchmark
        self.market = market
        self.infer_exp_name = f"{self.market}_MASTER_seed{self.seed}_backtest"

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
        for i, data in enumerate(data_loader):
            if i == 0:
                print(f"第一个batch的shape: {data.shape}")
            
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - 股票数量
            T - 回溯窗口长度，8
            F - 158个因子 + 63个市场信息 + 1个标签          
            '''
            # 提取特征和标签
            feature = data[:, :, 0:-1].to(self.device)  # 特征 [N, T, F-1]
            label = data[:, -1, -1].to(self.device)     # 标签 [N]
            
            if torch.any(torch.isnan(label)):
                print(f"警告：第{i}个batch包含NaN标签")
                continue

            pred = self._model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            # 反向传播和优化
            self.train_optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self.train_optimizer.step()

            if i % 10 == 0:  # 每10个batch打印一次进度
                print(f"已处理 {i} 个batch，当前loss: {loss.item():.6f}")

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
