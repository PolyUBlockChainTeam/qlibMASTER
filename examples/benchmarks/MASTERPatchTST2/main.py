#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces. 
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
import yaml
import argparse
import os
import pprint as pp
import numpy as np
import re
import logging
import datetime
from tqdm import tqdm

# 配置日志
def setup_logger(universe, only_backtest):
    """设置日志系统"""
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./backtest'):
        os.makedirs('./backtest')
    
    if only_backtest:
        log_file = f"./backtest/{universe}.log"
    else:
        log_file = f"./logs/{universe}.log"
    
    # 先移除 root logger 里所有的 handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建文件处理程序
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # 创建两个控制台处理程序
    stdout_handler = logging.StreamHandler(sys.stdout)  # 处理 INFO 及以上的日志
    stderr_handler = logging.StreamHandler(sys.stderr)  # 处理 WARNING 及以上的日志
    
    # 设置不同的日志级别
    stdout_handler.setLevel(logging.INFO)   # 只处理 INFO及以上
    stderr_handler.setLevel(logging.WARNING)  # 只处理 WARNING 及以上
    
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
    
    # 添加处理程序到root logger
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stdout_handler)
    logging.root.addHandler(stderr_handler)
    logging.root.setLevel(logging.INFO)  # 让 root logger 处理 INFO 及以上的日志
    
    # 返回root logger
    return logging.getLogger()

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_backtest", action="store_true", help="whether only backtest or not")
    parser.add_argument("--universe", type=str, default="csi300", choices=["csi300", "csi500"], 
                        help="set the market, you can choose `csi300` or `csi500`")
    return parser.parse_args()

def setup_directories():
    """创建必要的目录"""
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./backtest'):
        os.makedirs('./backtest')
    if not os.path.exists('./model'):
        os.makedirs('./model')

def update_config_file(universe):
    """更新配置文件中的市场参数"""
    config_file = "./workflow_config_master_patchtst2_Alpha158.yaml"  # 修改为PatchTST版本的配置文件
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 更新universe
    content = re.sub(r'csi\d+', universe, content)
    
    # 更新指数代码
    if universe == 'csi300':
        content = re.sub(r'SH\d+', 'SH000300', content)
    elif universe == 'csi500':
        content = re.sub(r'SH\d+', 'SH000905', content)
    
    with open(config_file, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    args = parse_args()
    
    # 创建必要的目录
    setup_directories()
    
    # 更新配置文件
    update_config_file(args.universe)
    
    # 设置日志系统
    logger = setup_logger(args.universe, args.only_backtest)
    logger.info(f"开始运行 MASTER-PatchTST2 模型 - universe: {args.universe}, only_backtest: {args.only_backtest}")
    logger.info(f"开始时间: {datetime.datetime.now()}")
    
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    logger.info("正在初始化数据...")
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    logger.info("正在加载配置文件...")
    with open("./workflow_config_master_patchtst2_Alpha158.yaml", 'r') as f:  # PatchTST2版本的配置文件
        config = yaml.safe_load(f)

    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = DIRNAME / f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}' \
                       f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    
    logger.info("正在预处理数据...")
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        logger.info(f'已保存预处理数据到 {h_path}')
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])

    ###################################
    # train model
    ###################################

    all_metrics = {
        k: []
        for k in [
            "IC",
            "ICIR",
            "Rank IC",
            "Rank ICIR",
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.information_ratio",
        ]
    }

    for seed in range(0, 3):
        logger.info("------------------------")
        logger.info(f"开始训练 seed: {seed}")

        config['task']["model"]['kwargs']["seed"] = seed
        model = init_instance_by_config(config['task']["model"])

        # start exp
        if not args.only_backtest:
            logger.info("开始训练模型...")
            # 重定向模型训练中的print输出到日志
            orig_print = print
            def custom_print(*args, **kwargs):
                msg = ' '.join(map(str, args))
                logger.info(msg)
                # orig_print(*args, **kwargs)  # 仍然保留控制台输出
            
            # 替换print函数
            import builtins
            builtins.print = custom_print
            
            # 使用tqdm显示训练进度
            with tqdm(total=model.n_epochs, desc=f"Training (seed={seed})") as pbar:
                def update_progress(epoch, train_loss, val_loss):
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.6f}',
                        'val_loss': f'{val_loss:.6f}'
                    })
                    pbar.update(1)
                
                # 修改模型的fit方法以支持进度显示
                model.fit(dataset=dataset, progress_callback=update_progress)
            
            # 恢复原始print函数
            builtins.print = orig_print
            logger.info(f"模型训练完成 (seed={seed})")
        else:
            logger.info(f"加载预训练模型 (seed={seed})...")
            model.load_model(f"./model/{config['market']}master_patchtst2_{seed}.pkl")  # PatchTST2版本的模型文件名

        logger.info("开始评估模型...")
        with R.start(experiment_name=f"workflow_seed{seed}"):
            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # Signal Analysis
            sar = SigAnaRecord(recorder)
            sar.generate()

            # backtest. If users want to use backtest based on their own prediction,
            # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
            par = PortAnaRecord(recorder, config['port_analysis_config'], "day")
            par.generate()

            metrics = recorder.list_metrics()
            logger.info(f"评估指标: {metrics}")
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            logger.info(f"所有指标: {all_metrics}")
    
    logger.info("------------------------")
    logger.info("最终评估结果:")
    for k in all_metrics.keys():
        logger.info(f"{k}: {np.mean(all_metrics[k])} +- {np.std(all_metrics[k])}")
    
    logger.info(f"结束时间: {datetime.datetime.now()}")
    logger.info("模型训练和评估完成")