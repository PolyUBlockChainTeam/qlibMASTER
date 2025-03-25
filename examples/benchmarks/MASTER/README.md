# MASTER - 市场引导型股票Transformer模型

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

## 📚 项目概述

这是 **MASTER**（Market-Guided Stock Transformer）模型的基准测试实现版本。MASTER是一个专为股票价格预测设计的Transformer模型，它结合了市场信息和个股信息，提高了预测准确性。

**论文**: [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/abs/2312.15235)

**原始代码**: [https://github.com/SJTU-Quant/MASTER](https://github.com/SJTU-Quant/MASTER)

## 🔧 环境配置

我们建议使用Conda来管理环境并运行代码。以下是设置环境的步骤：

> ⚠️ **注意**: 你需要自行安装PyTorch，建议使用支持CUDA的版本以加速训练。

### 自动配置

使用以下命令自动配置环境：

```bash
bash config.sh
```

该脚本会：
1. 创建名为`MASTER`的Conda环境（Python 3.12）
2. 安装必要的依赖
3. 配置qlib数据环境

### 手动配置

如果你希望手动配置环境，可以按照以下步骤进行：

```bash
# 创建Conda环境
conda create -n MASTER python=3.12
conda activate MASTER

# 安装依赖
pip install numpy
pip install --upgrade cython
cd ~/qlibMASTER
pip install -e .[dev]

# 初始化qlib
python -m qlib.install init
```

## 🚀 运行模型

### 使用脚本运行

我们提供了简便的脚本来运行模型，你可以在`run.sh`中设置：
- `universe`: 选择`csi300`（沪深300）或`csi500`（中证500）
- `only_backtest`: 设置为`true`只进行回测，设置为`false`先训练再回测

```bash
conda activate MASTER
bash run.sh
```

### 使用Python直接运行

你也可以直接使用Python运行模型，这样可以更灵活地设置参数：

```bash
conda activate MASTER
python main.py --universe csi300 --only_backtest  # 只在沪深300上进行回测
# 或者
python main.py --universe csi500  # 在中证500上进行训练和回测
```

## 💻 在PolyU HPC上运行

如果你在PolyU的高性能计算集群上运行，可以按照以下步骤操作：

### 1. 加载Slurm模块

```bash
module load slurm
```

### 2. 确保环境激活

```bash
conda activate MASTER
export PATH=/home/<YOUR_ID>/.conda/envs/MASTER/bin:$PATH
which pip  # 验证环境
```

### 3. 使用Slurm提交GPU任务

#### 方法1: 直接使用srun命令

```bash
srun --gres=gpu:7g.80gb:1 python main.py --universe csi300
```

#### 方法2: 创建批处理脚本

创建一个作业脚本`job_script.sh`：

```bash
#!/bin/bash
#SBATCH --job-name=master_model
#SBATCH --gres=gpu:7g.80gb:1  # 申请1个GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志
python main.py --universe csi300
```

提交作业：

```bash
sbatch job_script.sh
```

### 4. 管理作业

查看作业状态：
```bash
squeue -u <YOUR_ID>
```

取消作业：
```bash
scancel <JOB_ID>
```

## 📈 模型参数与配置

模型的主要配置参数在`workflow_config_master_Alpha158.yaml`文件中，包括：

- **训练轮数**: 40轮 (`n_epochs: 40`)
- **学习率**: 0.000008 (`lr: 0.000008`)
- **市场**: 默认为沪深300 (`market: csi300`)
- **数据周期**: 
  - 训练集: 2008-01-01 至 2014-12-31
  - 验证集: 2015-01-01 至 2016-12-31
  - 测试集: 2017-01-01 至 2020-08-01

## 📋 结果分析

训练完成后，模型会输出多种评估指标，包括：
- IC (Information Coefficient)
- ICIR (Information Coefficient Information Ratio)
- Rank IC
- Rank ICIR
- 年化收益率
- 信息比率

所有结果将保存在`logs`或`backtest`目录中，具体取决于运行模式。

## 🤝 贡献
louis
欢迎提交问题报告和改进建议。请随时提交Pull Request或创建Issue。