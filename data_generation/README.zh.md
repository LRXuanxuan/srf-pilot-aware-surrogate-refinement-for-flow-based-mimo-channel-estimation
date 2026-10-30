# CKM-SRF Unified Dataset Generator

该工程将原始 SRF CDL 数据、位置驱动 CKM/BIM 思路与高机动 WCM
先验统一到同一套 MATLAB 数据生成链中。工程不依赖 5G Toolbox，适合先做
方法原型和物理一致性验证；正式论文可以进一步将几何信道模块替换为
DeepMIMO、Sionna RT、Wireless InSite 或实测数据。

## 核心能力

- `H(rx,tx,subcarrier,snapshot,sample)` 统一信道维度；
- 64 发射、16 接收天线默认配置；
- 多场景固定散射体和阻挡物；
- 固定物理 `pathId` 的高移动宽带信道序列；
- 由独立历史样本统计构建 CKM，而不是读取测试瞬时真值；
- 空间块级 train/validation/test 划分；
- 双线性连续 CKM 查询；
- LoS、pathloss、角度-时延功率谱和 beam probability 先验；
- 正确的有限导频观测 `Y=H*P+N`；
- 可选 CFO/ICI 广义线性算子及其伴随算子；
- shard 分片保存，避免一次性占用过多内存；
- CKM-conditioned SRF、BIM/CGM 辅助任务所需标签。

## 快速开始

在 MATLAB 中执行：

```matlab
cd('C:/Users/DELL/Desktop/CKM_SRF_unified_generator')
run_demo
```

默认输出：

```text
output_demo/
  manifest.mat
  scene_01_ckm.mat
  scene_01_train_shard_001.mat
  scene_01_val_shard_001.mat
  scene_01_test_shard_001.mat
  ...
```

运行物理和 schema 测试：

```matlab
run('tests/run_all_tests.m')
```

## 正式实验配置

```matlab
addpath('configs')
cfg = research_config('D:/datasets/ckm_srf_64x16')
manifest = generate_ckm_srf_dataset(cfg)
```

正式生成前建议先将样本数缩小到 10，确认磁盘、内存和 MATLAB 版本兼容。

## 原始 SRF 与广义 CFO 模式

保持原 SRF 观测模型：

```matlab
cfg.cfo.enable = false;
```

此时每个子载波和快照满足：

```text
Y(:,:,k,t) = H(:,:,k,t) * P + N(:,:,k,t)
```

启用高移动 CFO/ICI：

```matlab
cfg.cfo.enable = true;
cfg.cfo.rangeHz = [-1500 1500];
```

此时 PBG 必须使用 `ckmsrf_apply_adjoint`，不能继续直接使用
`residual * P'` 作为完整梯度。

## 重要数据边界

- `truth` 只能作为监督标签或分析变量；
- 在线模型条件应来自 `state` 和 `context`；
- `context` 是从独立历史 CKM 插值得到，不包含测试瞬时 path gain；
- `aux` 是 BIM/beamforming 辅助标签，不应作为无误差 oracle 输入；
- 同一轨迹的全部快照始终属于同一空间 split。

详细字段见 [docs/SCHEMA.md](docs/SCHEMA.md)，SRF 接入方式见
[docs/SRF_INTEGRATION.md](docs/SRF_INTEGRATION.md)。

## Python/HDF5 导出

MATLAB `-v7.3` struct 对 Python 读取不够直接。工程提供显式 HDF5 导出：

```matlab
export_scene_to_hdf5( ...
  'output_demo/scene_01_ckm.mat', ...
  'output_demo_h5/scene_01_ckm.h5')

export_shard_to_hdf5( ...
  'output_demo/scene_01_train_shard_001.mat', ...
  'output_demo_h5/scene_01_train_shard_001.h5')
```

Python 示例位于 `python/example_read.py`，复数数组使用 `*_real` 和
`*_imag` 两个 dataset 保存，避免 MATLAB object reference 解码。

安装读取依赖并运行接口冒烟测试：

```bash
python -m pip install -r python/requirements.txt
python python/test_h5_loader.py \
  output_demo_h5/scene_01_train_shard_001.h5 \
  output_demo_h5/scene_01_ckm.h5
```

加载器会自动恢复 MATLAB HDF5 API 的轴顺序，Python 侧仍得到本文档约定的
`Nr x Nt x Nsc x T x N`，无需在训练代码中额外转置。
