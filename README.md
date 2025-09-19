# 基于共享提示与Mamba适配器的遥感图像文本检索方法<br>(A Remote Sensing Image Text Retrieval Method Based on the Shared Prompt and Mamba Adapter)



## 环境安装 (Installation)
1.  **创建一个新的 Conda 环境**
    ```bash
    conda create -n SPMA python=3.8 
    conda activate SPMA
    ```

2.  **安装核心依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **安装 Mamba 和 Causal-Conv1D**
    安装与环境中 PyTorch 和 CUDA 版本相匹配的预编译版本。

    *   **causal-conv1d**: 从 [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases) 页面下载 `1.4.0` 版本的 wheel 文件 
        ```bash
        pip install /path/to/your/downloaded/causal_conv1d-xxx.whl
        ```

    *   **mamba-ssm**: 从 [mamba-ssm releases](https://github.com/state-spaces/mamba/releases) 页面下载 `2.2.2` 版本的 wheel 文件。
        ```bash
        pip install /path/to/your/downloaded/mamba_ssm-xxx.whl
        ```
    > **注意**: 请下载对应的 `cxx11abiFALSE` 版本。

4.  **编译 `GrootVL` 依赖**
    ```bash
    # 编译 TreeScan
    cd GrootVL/GrootV/third-party/TreeScan
    pip install -v -e .

    # 编译 TreeScanLan
    cd GrootVL/GrootL/third-party/TreeScanLan
    pip install -v -e .
    ```

## 数据准备 (Data Preparation)

1.  **下载数据集**
    本项目使用了 RSICD 和 RSITMD 数据集。我们提供了百度网盘下载链接：
    > 链接: https://pan.baidu.com/s/1ERBGIOzDQOxV450U3-ufjw?pwd=5m3n (提取码: 5m3n)

2.  **放置数据集**
    下载并解压后，请将图像文件夹放置在 `./data/images/` 目录下，目录结构应如下所示：
    ```
    SPMA/
    ├── data/
    │   ├── images/
    │   │   ├── rsicd/
    │   │   │   ├── image1.jpg
    │   │   │   └── ...
    │   │   └── rsitmd/
    │   │       ├── image1.jpg
    │   │       └── ...
    │   └── finetune/
    │       ├── rsicd_train.json
    │       └── ...
    └── ...
    ```

## 训练 (Training)

我们提供了单卡和多卡（数据并行）的训练脚本。

### RSICD 数据集

*   **单卡训练**:
    ```bash
    python run.py --task itr_rsicd_vit --dist gpu0 --config ./configs/Retrieval_rsicd_vit.yaml --output_dir ./output/train/rsicd
    ```

*   **多卡训练** (例如，使用 4 个 GPU):
    ```bash
    python run.py --task itr_rsicd_vit --dist f4 --config ./configs/Retrieval_rsicd_vit.yaml --output_dir ./output/train/rsicd
    ```

### RSITMD 数据集

*   **单卡训练**:
    ```bash
    python run.py --task itr_rsitmd_vit --dist gpu0 --config ./configs/Retrieval_rsitmd_vit.yaml --output_dir ./output/train/rsitmd
    ```

*   **多卡训练**:
    ```bash
    python run.py --task itr_rsitmd_vit --dist f4 --config ./configs/Retrieval_rsitmd_vit.yaml --output_dir ./output/train/rsitmd
    ```

## 评估 (Evaluation)

我们提供了预训练好的模型权重用于快速评估。

1.  **下载权重**
    > 链接: https://pan.baidu.com/s/15SO0bu2JGahZY0usszwnYw?pwd=zs1c (提取码: zs1c)

2.  **放置权重**
    下载并解压后，将权重文件和对应的配置文件放入 `./checkpoint/` 目录下，目录结构应如下所示：
    ```
    SPMA/
    ├── checkpoint/
    │   ├── rsicd/
    │   │   ├── checkpoint_best.pth
    │   │   └── config.yaml
    │   └── rsitmd/
    │       ├── checkpoint_best.pth
    │       └── config.yaml
    └── ...
    ```

3.  **运行评估脚本**
    评估通常在单 GPU 上进行。

    *   **在 RSICD 上评估**:
        ```bash
        python run.py --task itr_rsicd_vit --dist gpu0 --config ./checkpoint/rsicd/config.yaml --output_dir ./output/test/rsicd --checkpoint ./checkpoint/rsicd/checkpoint_best.pth --evaluate
        ```

    *   **在 RSITMD 上评估**:
        ```bash
        python run.py --task itr_rsitmd_vit --dist gpu0 --config ./checkpoint/rsitmd/config.yaml --output_dir ./output/test/rsitmd --checkpoint ./checkpoint/rsitmd/checkpoint_best.pth --evaluate
        ```

## 致谢 (Acknowledgements)
*   [HarMA](https://github.com/seekerhuang/HarMA)
*   [GrootVL](https://github.com/EasonXiao-888/MambaTree)
*   [Hydra](https://github.com/goombalab/hydra)




