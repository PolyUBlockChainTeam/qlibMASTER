# 检查MASTER环境是否已存在
if conda info --envs | grep -q "MASTER"; then
    echo "MASTER环境已存在，跳过创建步骤..."
else
    # 创建新的conda环境
    echo "创建新的MASTER环境..."
    conda create -n MASTER python=3.12
fi

# 激活MASTER环境
eval "$(conda shell.bash hook)"
conda activate MASTER

# 检查当前环境是否为MASTER
if [[ $CONDA_DEFAULT_ENV != "MASTER" ]]; then
    echo "当前环境不是MASTER，正在激活MASTER环境..."
    conda activate MASTER
fi

# 检查pip路径并设置正确的PATH
HOME_PATH=$(echo ~)
EXPECTED_PIP_PATH="$HOME_PATH/.conda/envs/MASTER/bin/pip"
CURRENT_PIP_PATH=$(which pip)

if [[ $CURRENT_PIP_PATH != $EXPECTED_PIP_PATH ]]; then
    echo "设置正确的PATH环境变量..."
    export PATH="$HOME_PATH/.conda/envs/MASTER/bin:$PATH"
fi

# install `qlib`
pip install numpy
pip install --upgrade cython

# 检查qlibMASTER目录是否存在
if [ -d ~/qlibMASTER ]; then
    echo "qlibMASTER目录已存在，进入该目录..."
    cd ~/qlibMASTER
else
    echo "qlibMASTER目录不存在，克隆qlib仓库..."
    cd ~ && git clone https://github.com/PolyUBlockChainTeam/qlibMASTER.git qlibMASTER
    cd ~/qlibMASTER
fi

# 无论目录是否存在，都执行pip install
echo "安装qlib开发版本..."
pip install -e .[dev]
cd -

# 不再需要手动执行pip install
# echo "请手动进行cd ~/qlibMASTER && pip install -e .[dev]"

python -m qlib.install init
pip install -r requirements.txt

# the following codes copy the `*.so` files in the installed `qlib` package to our customizable `qlib`
# path=$(which conda)
# parent_path=$(dirname $path)
# parent_path=$(dirname $parent_path)
# lib_path="$parent_path/envs/MASTER/lib/python3.12/site-packages/qlib/data/_libs"
# echo $lib_path
# find $lib_path -type f -name "*.so" -exec cp {} "../../../qlib/data/_libs" \;

# we directly use our customizable `qlib`
# pip uninstall pyqlib