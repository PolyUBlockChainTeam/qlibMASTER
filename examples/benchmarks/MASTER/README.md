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

## 多人协作开发
**1. 获取最新代码**：<br>
在开始任何修改前，确保本地代码是最新的：


```bash
git fetch origin
git pull origin main
```

**2. 提交本地更改**：
```bash
git add .
git commit -m "Your commit message"
```


**3. 推送更改到远程仓库**：
```bash
git push origin main
```

**4. 处理冲突**：<br>
如果远程仓库有其他人的修改，而你未同步就进行了提交，可能会出现冲突。解决冲突步骤如下：<br>
<br>
拉取远程更新并尝试自动合并：
```bash
git pull origin main
```

如果有冲突，Git 会提示冲突文件，打开冲突文件，你会看到类似以下标记：
```diff
<<<<<<< HEAD
你的代码
=======
对方的代码
>>>>>>> 对方提交的commit-id
```

手动修改冲突部分，保留需要的代码。<br>
**标记冲突已解决**：
```bash
git add <conflict_file>
```

**再次提交**：
```bash
git commit -m "Resolve merge conflict"
```
推送解决后的代码：
```bash
git push origin main
```
**5. 分支管理（推荐）**：<br>
为了降低冲突风险，建议每个开发者在自己的分支开发：

```bash
git checkout -b feature/your-feature
```
开发完成后，先将主分支的更新合并到自己的分支：

```bash
git checkout feature/your-feature
git merge main
```
解决冲突后推送分支代码：

```bash
git push origin feature/your-feature
```
通过 Pull Request 提交分支合并到主分支，确保审核后再合并。

**忽略 data 文件夹**<br>
如果不希望 data 文件夹被推送到 GitHub，请按照以下步骤操作：<br>

**创建或编辑 .gitignore 文件，添加以下内容**：

```bash
data/
```
**移除已被 Git 跟踪的 data 文件夹**：

```bash
git rm -r --cached data
```
**提交更改并推送**：

```bash
git add .gitignore
git commit -m "Ignore data folder and remove from Git tracking"
git push origin main
```
从此，data 文件夹将不再被 Git 跟踪，并不会推送到 GitHub。<br>

**忽略 data.json 文件**<br>
如果不希望 data.json 文件被推送到 GitHub，请按照以下步骤操作：<br>

**创建或编辑 .gitignore 文件，添加以下内容**：

```bash
data.json
```
**移除已被 Git 跟踪的 data.json 文件**：

```bash
git rm -r --cached data.json
```
**提交更改并推送**：

```bash
git add .gitignore
git commit -m "Ignore data.json and remove from Git tracking"
git push origin main
```
从此，data.json 文件将不再被 Git 跟踪，并不会推送到 GitHub。<br>
<br>
小贴士<br>
随时同步远程仓库：避免提交较大修改后才同步，这样会增加冲突概率。<br>
小步提交：更频繁地提交修改，减少冲突范围。<br>
定期代码审查：通过 Pull Request 进行代码合并时，便于团队发现潜在问题。<br>

## 🤝 贡献

欢迎提交问题报告和改进建议。请随时提交Pull Request或创建Issue。
