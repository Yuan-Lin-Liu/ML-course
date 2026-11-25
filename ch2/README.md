# 操作指令

在项目根目录的终端执行。需要 uv；当前环境锁定为 Windows x64 + NVIDIA CUDA。

## 安装环境

```powershell
uv python install 3.12
uv sync --locked
```

## 放置数据

```text
data/
├── 2-MedImage-TrainSet/
│   ├── disease/  # 646张
│   └── normal/   # 993张
└── 2-MedImage-TestSet/
    ├── disease/  # 100张
    └── normal/   # 150张
```

## 训练与输出

```powershell
# 训练20轮
uv run --locked python train.py

# 训练40轮
uv run --locked python train.py --epochs 40

# 修改参数，可选早停（默认不开启）
uv run --locked python train.py --epochs 40 --batch-size 32 --lr 0.0001 --patience 8
```

`train.py` 独立完成训练、验证、保存最佳模型和loss/准确率曲线。训练阶段仅检查测试集是否与训练集重复，不计算测试指标。结果位于 `outputs/resnet18_时间戳/`，打开 `results.html` 查看。重新运行会新建目录，不覆盖旧结果；不是接着旧模型续训。

## 测试与输出

```powershell
# 最近完成训练的模型，随机抽取80%测试图像
uv run --locked python evaluate.py

# 固定抽样，方便复现
uv run --locked python evaluate.py --fraction 0.8 --seed 42

# 评估全部测试图像
uv run --locked python evaluate.py --fraction 1

# 指定已有模型
uv run --locked python evaluate.py --checkpoint "outputs/实验目录/best.pt" --fraction 1
```

结果位于 `outputs/test80_时间戳/`，包含指标、逐张预测、抽样清单、ROC/PR曲线、混淆矩阵和 `results.html`。`test80` 是评估目录前缀，实际抽样比例以 `--fraction` 为准。患病为正类，阈值0.5；测试loss为无权重平均BCE，与加权训练loss区分。

