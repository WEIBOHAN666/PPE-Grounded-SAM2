# 模型推理环境

本文说明如何安装官方模型依赖，再把单图预测接入关联与复查流程。以下命令在项目根目录执行，使用 Linux / WSL Bash。

## 环境准备

基础工具与 GPU 推理使用独立环境。建议用 Python 3.10 或 3.11 创建模型环境，并根据服务器驱动选择匹配的 PyTorch / CUDA 组合；这是一条待实际 GPU 验证的安装路线。

SAM 2 官方要求 Python ≥ 3.10、PyTorch ≥ 2.5.1、TorchVision ≥ 0.20.1；Windows 用户建议使用 WSL。CUDA 扩展编译还需要匹配的 CUDA Toolkit、`nvcc` 和 C++ 编译工具。详见 [SAM 2 安装说明](https://github.com/facebookresearch/sam2#installation) 与 [Grounding DINO 官方仓库](https://github.com/IDEA-Research/GroundingDINO)。

```bash
python3.11 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

先在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择系统与 CUDA 版本，执行页面给出的 `torch` / `torchvision` 安装命令。不要直接复制其他机器的 CUDA 版本号。

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
nvcc --version
```

GPU 推理前确认 CUDA 可用，`CUDA_HOME` 指向实际 Toolkit 安装目录，且编译工具链与 PyTorch 使用的 CUDA 兼容。

## 安装模型源码

```bash
mkdir -p third_party checkpoints
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
python -m pip install --no-build-isolation -r requirements-models.txt
python -m pip check
python main.py doctor
```

先安装 PyTorch，再安装模型包：Grounding DINO 的构建过程会导入 PyTorch。这里关闭构建隔离以使用已配置的 PyTorch 环境；构建工具需要预先安装。安装失败时按上游说明排查，不应把基础工具测试通过视为 CUDA 扩展已可用。

当前采用上游源码安装，未锁定两个仓库的 commit。复现实验时应记录实际依赖与源码版本：

```bash
mkdir -p outputs/environment
python -m pip freeze > outputs/environment/pip-freeze.txt
git -C third_party/GroundingDINO rev-parse HEAD > outputs/environment/groundingdino-commit.txt
git -C third_party/sam2 rev-parse HEAD > outputs/environment/sam2-commit.txt
```

## 下载权重

从官方发布渠道取得权重，保存为：

```text
checkpoints/
├── groundingdino_swint_ogc.pth
└── sam2.1_hiera_small.pt
```

- [Grounding DINO Swin-T 权重](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- [SAM 2.1 权重与配置对应关系](https://github.com/facebookresearch/sam2#download-checkpoints)

本项目默认 SAM 2.1 Small 的配置名为 `configs/sam2.1/sam2.1_hiera_s.yaml`，它由官方 SAM 2 包提供，不是项目根目录下的配置文件。

## 单图推理

将自备原图保存到 `data_samples/example.jpg`：

```bash
python main.py infer \
  --image data_samples/example.jpg \
  --gdino-config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino-checkpoint checkpoints/groundingdino_swint_ogc.pth \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint checkpoints/sam2.1_hiera_small.pt \
  --prompt "person. head. helmet." \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda \
  --output-dir outputs/inference/example
```

输出 `prediction.json`、`overlay.jpg` 和 `masks/`。检测阈值与后处理阈值属于不同阶段：关联及复查入口还会按 `detection_score >= 0.6` 筛选实例。

## 关联与复查

```bash
python scripts/check_prediction_association.py \
  --prediction outputs/inference/example/prediction.json \
  --output-dir outputs/review/example
```

将结果下载到另一台机器后，如果 JSON 中的图片路径不再有效，为单图补充 `--image 本地原图路径`。输出目录须不存在或为空。

完整数据结构见 [数据与关联约定](DATA_FORMAT.md)。请先检查真实检测中的类别语义及置信度分布，再评估关联效果；本项目尚未提供经过真实权重验证的精度或延迟指标。

