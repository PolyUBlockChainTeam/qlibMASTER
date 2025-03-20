#!/bin/bash
#SBATCH --job-name=master_model
#SBATCH --gres=gpu:7g.80gb:1  # 申请1个GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志
python main.py --universe csi300