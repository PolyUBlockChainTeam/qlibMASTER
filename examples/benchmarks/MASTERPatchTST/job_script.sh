#!/bin/bash
#SBATCH --job-name=master_model
#SBATCH --gres=gpu:4g.40gb:1  # 申请GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志

# #SBATCH --gres=gpu:7g.80gb:1  # 申请GPU 
# #SBATCH --gres=gpu:2g.20gb:2  # 申请GPU 
# #SBATCH --gres=gpu:4g.40gb:1  # 申请GPU 

python main.py --universe csi300