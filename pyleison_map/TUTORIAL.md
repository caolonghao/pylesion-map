# pyleison_map 教程（dTLVC 版病灶-症状建模）

本教程介绍 `pyleison_map` 目录下的两个核心模块：
- `lesion_models.py`：病灶-症状建模框架（支持回归/分类、dTLVC 预处理、beta/importance 图导出）
- `cv_utils.py`：交叉验证与网格搜索辅助工具

教程将覆盖环境准备、数据格式、快速上手示例、交叉验证/网格搜索、beta 图导出与可视化、常用参数以及调优建议。

---

## 环境准备

- Python 3.8+（推荐 3.9 或更高）
- 依赖包：
  - `numpy`, `pandas`, `scikit-learn`
  - 可选：`SimpleITK`（用于读取 NIfTI/医学影像；也可使用已在内存的 `numpy` 3D 数组）
  - 可选：`scipy`（用于相关系数计算）
- 安装示例：

```bash
pip install numpy pandas scikit-learn SimpleITK scipy
```

- 代码导入方式：位于 `pyleison_map` 目录下时可直接：

```python
from pyleison_map import lesion_models
from pyleison_map import cv_utils
```

说明：`cv_utils.py` 中有一段为笔记本环境准备的路径注入（将 `/mnt/data` 加入 `sys.path` 并以模块名 `lesion_models` 导入）。在当前项目结构下，推荐按上面的包导入方式使用；如你在独立脚本中运行且导入失败，可临时：

```python
import sys
sys.path.insert(0, "/Users/tonycao/mycode/mutism/lesion_map/pyleison_map")
import lesion_models  # 或 from pyleison_map import lesion_models
```

---

## 数据格式与 dTLVC 简介

- 输入影像：列表 `imgs`，每个元素为 3D 病灶体积（标准空间对齐后）。
  - 类型：`SimpleITK.Image` 或 `numpy.ndarray`（形状为 `(Z, Y, X)`）
  - 病灶体积通常是二值 mask（病灶位置为 1/正值）
- 目标标签 `y`：
  - 回归：连续值数组，如症状评分
  - 分类：类别标签数组（字符串或整型均可）

- dTLVC（direct Total Lesion Volume Control）：在特征向量化后，对每个样本行进行 L2 单位范数归一化，用以显式控制总病灶体积的影响。
  - 在 `lesion_models` 的预处理器 `LesionPreprocessor` 中实现，`fit_transform` 会：
    - 根据设定的脑区/联合掩膜与最小病灶计数阈值构造有效体素掩膜
    - 将 3D 体积在掩膜上展平为特征向量
    - 可根据 `keep_empty_subjects` 决定是否剔除在掩膜后仍全零的样本
    - 对每个样本向量做单位范数归一化（dTLVC）

---

## 代码结构速览：预处理与模型拆分

为了提升可维护性，`lesion_models` 已拆分为独立子模块，按职责组织在 `pyleison_map.models.lesion_ml` 包内：

- `preprocessing.py`：提供 `LesionPreprocessor` 与 `preprocessing_kwargs_from_options`。  
  - `LesionPreprocessor` 负责掩膜构建、向量化、可选的 voxelwise z-score 与 dTLVC，并记录 `kept_idx_` 等状态以支持体素重建。  
  - `preprocessing_kwargs_from_options` 将高层的 `PreprocessOptions`（出自 `pyleison_map.preprocess.lesion_matrix`）转换为模型工厂所需的关键字参数，供 orchestrator、交叉验证等调用方复用。
- `base.py`：定义 `LesionModelBase`，封装通用的 `fit/predict/beta_map` 接口与持久化钩子。
- `models.py`：具体的 sklearn / xgboost 包装类、`make_model` 工厂以及 `MODEL_REGISTRY`。
- `persistence.py`：统一的 `save_model` / `load_model` 帮助函数，负责写入/恢复 estimator、掩膜与标签编码器等状态。

旧文件 `pyleison_map.models.lesion_ml_models` 仍作为兼容层存在，会直接 re-export 以上符号；因此原有代码仍可以使用 `from pyleison_map import lesion_models`。如果希望更细粒度地控制预处理或持久化流程，可直接从新子包导入所需组件，例如：

```python
from pyleison_map.models.lesion_ml import LesionPreprocessor, make_model
```

后续示例依旧使用 `lesion_models` 命名空间，但了解其内部拆分后，你可以在需要时组合底层模块实现更复杂的工作流。

---

## 预处理工具箱：`pyleison_map.preprocess`

除模型内部的向量化步骤外，`preprocess` 目录还提供了若干常用的影像处理工具，方便在建模前完成配准、重采样与矩阵化：

- `lesion_matrix.py`  
  - `build_lesion_matrix(images, ...)`：将一批病灶体积转换为 subject-by-voxel 矩阵，支持最小病灶计数、掩膜、voxelwise z-score 与行归一化等选项，返回 `LesionMatrixResult`（包含 `matrix`, `feature_mask`, `kept_indices` 等元数据）。  
  - `vectorize_image_to_mask(image, feature_mask, volume_shape, ...)`：使用已有的特征掩膜将单个体积转换为向量，常用于推理或持久化后的载入。  
  - `PreprocessOptions`：面向 orchestrator 的高层配置对象，便于在不同流程间共享参数。
- `normalisation.py`  
  - `normalize_to_template(images, template, ...)`：基于 antspy 进行图像配准，可一次性处理影像与对应分割，选择 SyN 等变换，并返回 `NormalizationResult`（包含配准后的影像、变换路径等信息）。  
  - 需要提前安装 `antspyx`；若设置 `save_registered_images/segmentations=True`，记得提供输出目录。
- `resample.py`  
  - `resample_images(images, target_spacing=None, target_shape=None, ...)`：同样依赖 antspy，按目标体素间距或体素数重采样，支持保存到磁盘并返回 `ResampleResult` 以追踪输出路径和空间信息。

这些函数可单独使用。例如，在将原始病灶影像送入模型前，可先调用 `normalize_to_template` 完成配准，再用 `resample_images` 统一分辨率，最后通过 `build_lesion_matrix` 生成矩阵/掩膜供后续分析或机器学习模型复用。

```python
from pathlib import Path

from pyleison_map.preprocess import normalisation, resample, lesion_matrix, PreprocessOptions

# 1) 配准到模板空间
template_path = Path("/data/template/MNI.nii.gz")
norm_results = normalisation.normalize_to_template(
    images=["/data/raw/sub01.nii.gz", "/data/raw/sub02.nii.gz"],
    template=template_path,
    save_registered_images=True,
    image_output_dir="/data/registered",
)
registered_imgs = [res.warped_image for res in norm_results]

# 2) 可选：统一分辨率/体素数
resampled = resample.resample_images(
    registered_imgs,
    target_spacing=(1.0, 1.0, 1.0),
    interpolator="linear",
)
resampled_imgs = [res.resampled_image for res in resampled]

# 3) 构建病灶矩阵并获取掩膜、保留索引等信息
pre_opts = PreprocessOptions(min_voxel_lesion_count=3, apply_subjectwise_l2=True)
matrix_result = lesion_matrix.build_lesion_matrix(
    resampled_imgs,
    mask=pre_opts.mask,
    binarize=pre_opts.binarize,
    min_voxel_lesion_count=pre_opts.min_voxel_lesion_count,
    drop_empty_rows=pre_opts.drop_empty_rows,
    apply_voxelwise_zscore=pre_opts.apply_voxelwise_zscore,
    apply_subjectwise_l2=pre_opts.apply_subjectwise_l2,
)
print(matrix_result.matrix.shape, matrix_result.feature_mask.sum())
```

---

## 快速上手：回归与分类

### 示例 1：回归（Lasso）

```python
from pyleison_map import lesion_models
import SimpleITK as sitk
import numpy as np

# 1) 读取病灶影像（举例：从 nifti 文件加载）
paths = ["/path/to/sub01_lesion.nii.gz", 
          "/path/to/sub02_lesion.nii.gz", 
          # ...
         ]
imgs = [sitk.ReadImage(p) for p in paths]

# 2) 回归目标（例如行为评分）
y = np.array([12.5, 8.0, # ..., 
])

# 3) 构建模型（Lasso 回归）
model = lesion_models.make_model(
    kind="lasso",
    task="regression",
    min_lesion_count=3,       # 至少有 3 个受试者在该体素有病灶才纳入
    brain_mask=None,          # 使用病灶联合掩膜作为分析范围
    keep_empty_subjects=True, # 保留掩膜后仍全零的样本（如无病灶样本）
)

# 4) 训练
model.fit(imgs, y, binarize=True)

# 5) 预测（单个影像）
pred = model.predict(imgs[0])
print("Predicted score:", pred)

# 6) 导出 beta 图（系数重建为 3D 体素图）
beta_vol = model.beta_map()  # numpy.ndarray shape 与原体积一致
```

### 示例 2：分类（线性 SVM）

```python
from pyleison_map import lesion_models
import numpy as np

# imgs: 同上，列表形式（numpy 或 SimpleITK）
labels = np.array(["aphasia", "aphasia", "no_aphasia", # ...
])

model = lesion_models.make_model(
    kind="linear_svm",
    task="classification",
    min_lesion_count=2,
    signed_importance_for_trees=False,  # 对树模型生效；线性 SVM 返回线性权重
)

model.fit(imgs, labels, binarize=True)
y_pred = np.array([model.predict(im) for im in imgs])
print("Training-set predictions:", y_pred)

beta = model.beta_map()
# 线性多分类的 beta_map 返回 dict: 类别 -> 3D 权重图
# 若是二分类或单向量，则返回单个 3D numpy 数组
print(type(beta), "keys: " if isinstance(beta, dict) else "volume shape:", 
      list(beta.keys()) if isinstance(beta, dict) else beta.shape)
```

---

## 可用模型种类（`make_model(kind, task, **kwargs)`）

- `kind` 可取：
  - `lasso`（仅回归）
  - `linear_svr`（仅回归，线性核 SVR）
  - `svr_rbf`（仅回归，RBF 核 SVR，支持近似 beta 图）
  - `linear_svm`（仅分类）
  - `logistic_l1`（仅分类，L1 正则逻辑回归）
  - `rf`（回归/分类：随机森林）
  - `ada`（回归/分类：AdaBoost）
  - `xgb`（回归/分类：XGBoost，如可用）

- 重要公共参数（在所有模型的构造函数中通过 `**kwargs` 传入）：
  - `min_lesion_count: int` 有效体素最小病灶计数阈值（默认 1）
  - `brain_mask: Optional[np.ndarray]` 3D 布尔掩膜（若为空，将使用病灶联合掩膜）
  - `keep_empty_subjects: bool` 是否保留掩膜后全零的样本（默认 True）
  - `signed_importance_for_trees: bool` 树/Boosting 模型的 importance 可选取签名版本（默认 False）

- 具体估计器的超参数：
  - 例如 `LassoRegression(alpha=1.0)`、`LinearSVR(C=1.0)`、`SVR_RBF_Model(C=30.0, gamma=5.0, epsilon=0.1)`、`LinearSVMClassifier(C=1.0)` 等；
  - 分类模型会使用 `LabelEncoder` 自动编码标签。

---

## 交叉验证（`cv_utils.cross_validate_lesion_model`）

用于评估单一配置在 K 折上的表现。自动根据任务类型选择 `KFold` 或 `StratifiedKFold`。

```python
from pyleison_map import cv_utils
import numpy as np

out = cv_utils.cross_validate_lesion_model(
    imgs=imgs,
    y=np.asarray(y),
    kind="lasso",
    task="regression",
    base_kwargs={
        "min_lesion_count": 3,
        "brain_mask": None,
        "keep_empty_subjects": True,
        "signed_importance_for_trees": False,
    },
    n_splits=5,
    random_state=42,
    shuffle=True,
)

print("每折指标:", out["per_fold"])    # list[dict]
print("平均指标:", out["mean"])       # dict
```

- 回归指标包含：`r2`, `rmse`, `mae`, `mse`, `pearson_r`, `spearman_r`
- 分类指标包含：`acc`, `balanced_acc`, `f1`, `roc_auc`, `pr_auc`, `log_loss`（按可用性）

---

## 顶层 orchestration 快速体验

若希望“一次调用”完成影像→矩阵→模型/统计分析，推荐使用 `pyleison_map.analysis` 模块，它提供三个入口函数：

| 函数 | 功能 | 典型输出 |
| --- | --- | --- |
| `run_ml_model` | 训练机器学习模型（Lasso、SVR、RF 等） | `MLModelResult`，包含预处理配置、保留样本索引与已训练 `LesionModelBase` |
| `run_statistical_model` | 仅 SCCAN：统计驱动且可预测 | `StatisticalModelResult`，含预处理信息、保留索引与 `SCCANResult` |
| `run_statistical_analysis` | 纯统计检验（Chi-square、t-test/Welch、Brunner-Munzel） | `StatisticalAnalysisResult`，携带预处理信息、保留索引与相应结果 dataclass |

所有入口都接受统一的影像列表与标签列表输入：

```python
img_list = [img1, img2, ...]  # 可为 NIfTI 路径、SimpleITK/ANTs 对象或 numpy 数组
y_list = [y1, y2, ...]        # 连续值或类别标签
```

### 1. 训练 ML 模型

```python
from pyleison_map.analysis import run_ml_model

ml_result = run_ml_model(
    images=img_list,
    behavior=y_list,
    method="lasso",                 # 也可用 "linear_svr", "rf", "svr_rbf" 等
    task="regression",              # 或 "classification"
    preprocess=PreprocessOptions(    # 可选；未传入时使用模型默认 dTLVC 设置
        min_voxel_lesion_count=5,
        drop_empty_rows=False,
        apply_voxelwise_zscore=False,
        apply_subjectwise_l2=True,
    ),
    model_kwargs={
        # 仍可覆盖模型自身超参数，例如 Lasso 的 alpha
        "alpha": 1.0,
    },
)

trained_model = ml_result.model
print(trained_model.predict(img_list[0]))
trained_model.save("artifacts/ml/lasso_model")
```

`run_ml_model` 返回的 `LesionModelBase` 完整支持 `predict`、`beta_map`、`save/load` 等接口。

### 2. SCCAN（统计模型+预测）

```python
from pyleison_map.analysis import run_statistical_model, PreprocessOptions

pre_opts = PreprocessOptions(
    min_voxel_lesion_count=5,
    apply_voxelwise_zscore=True,
    apply_subjectwise_l2=False,
)

sccan_result = run_statistical_model(
    images=img_list,
    behavior=y_list,
    preprocess=pre_opts,
    sccan_kwargs={
        "optimize_sparseness": True,
        "p_threshold": 0.05,
    },
)

print(sccan_result.result.cca_correlation)
print(sccan_result.result.statistic.shape)  # 病灶权重向量
print(sccan_result.kept_indices.shape)      # 经过预处理后保留下的样本
```

### 3. 纯统计检验

```python
from pyleison_map.analysis import run_statistical_analysis, PreprocessOptions

analysis_pre = PreprocessOptions(
    min_voxel_lesion_count=3,
    apply_voxelwise_zscore=False,
    apply_subjectwise_l2=False,
)

chi_out = run_statistical_analysis(
    images=img_list,
    behavior=y_list,           # chi-square 需二分类标签
    method="chi_square",
    preprocess=analysis_pre,
    method_kwargs={"yates_correction": True, "n_permutations": 1000},
)

chi_result = chi_out.result
print(chi_result.statistic[:10], chi_result.pvalue[:10])
```

若 `method="ttest"` 或 `"welch"` 会返回 `TTestResult`；`"brunner_munzel"` 返回 `BrunnerMunzelResult`，可进一步指定 voxelwise permutation、FWER rank 等参数以复现 LESYMAP 行为。

---

## 网格搜索（`cv_utils.grid_search_lesion_model`）

支持串行网格搜索，参数命名约定：
- 以 `est__` 前缀表示估计器参数（如 `est__C`, `est__alpha`）
- 以 `prep__` 前缀表示预处理器属性（如 `prep__min_lesion_count`）
- 其他顶层属性（如 `signed_importance_for_trees`）可直接设置

示例（回归，选择 Lasso 的 `alpha` 和预处理器的最小病灶计数）：

```python
from pyleison_map import cv_utils
import numpy as np

grid_out = cv_utils.grid_search_lesion_model(
    imgs=imgs,
    y=np.asarray(y),
    kind="lasso",
    task="regression",
    base_kwargs={"keep_empty_subjects": True},  # 基础设置，可选
    param_grid={
        "est__alpha": [0.1, 1.0, 5.0],
        "prep__min_lesion_count": [1, 3],
    },
    primary_metric="r2",    # 若不指定：回归默认 r2；分类默认 acc
    n_splits=5,
    refit=True,              # 自动用最佳参数在全数据上拟合 best_model
)

print("CV 结果表:")
print(grid_out["cv_results"])      # pandas.DataFrame，各参数组合的平均指标
print("最佳参数:", grid_out["best_params"]) 
print("最佳分数:", grid_out["best_score"]) 
best_model = grid_out["best_model"]   # 若 refit=True
```

分类示例（线性 SVM，调 `C` 与是否保留空样本）：

```python
grid_out = cv_utils.grid_search_lesion_model(
    imgs=imgs,
    y=labels,
    kind="linear_svm",
    task="classification",
    base_kwargs={},
    param_grid={
        "est__C": [0.1, 1.0, 10.0],
        "prep__keep_empty_subjects": [True, False],
    },
    primary_metric="balanced_acc",  # 分类可选：acc, balanced_acc, f1, roc_auc, pr_auc, log_loss
)
```

---

## beta/importance 图导出与可视化

- 训练后调用 `model.beta_map()`：
  - 线性模型（Lasso/LinearSVR/LinearSVC/Logistic）：返回系数重建成的 3D 体积
  - 树/Boosting（RF/Ada/XGB）：返回特征重要性重建体积；若 `signed_importance_for_trees=True` 则可返回带符号版本
  - 非线性 RBF SVM/SVR：返回近似的“敏感度”/preimage 权重图（用于可视化和解释）

- 可视化示例（保存为 NIfTI）：

```python
import SimpleITK as sitk
import numpy as np

beta_vol = model.beta_map()            # numpy.ndarray, shape == 原体积形状
beta_img = sitk.GetImageFromArray(beta_vol.astype(np.float32))

# 若需要复制空间信息（仿射/头信息），从某个模板影像拷贝：
tpl = imgs[0] if isinstance(imgs[0], sitk.Image) else None
if tpl is not None:
    beta_img.CopyInformation(tpl)

sitk.WriteImage(beta_img, "/path/to/output_beta.nii.gz")
```

---

## 常用参数与建议

- `min_lesion_count`
  - 提高该阈值能过滤掉极少出现病灶的体素，减小噪声；但过高可能减少信息
- `brain_mask`
  - 若已知分析感兴趣区（如灰质掩膜或束带），可传入 3D 布尔掩膜以限制范围
- `keep_empty_subjects`
  - 若为 `False`，掩膜后仍为全零的样本会被剔除；适用于需要严格控制空样本的情境
- `binarize`
  - 默认 `True`，适用于病灶掩膜；若输入已为加权图，可改为 `False`
- 分类评估指标选择
  - 非均衡数据集推荐使用 `balanced_acc`, `roc_auc`, `pr_auc`
- 树模型解释性
  - importance 为相对贡献；对于方向性（正/负）解释，线性模型更直接

---

## 常见问题（FAQ）

- Q：我的数据是 `numpy` 数组，不是 `SimpleITK.Image`，可以吗？
  - A：可以。只需保证每个样本是 3D 数组 `(Z, Y, X)`，与其他样本在同一标准空间。`fit`/`predict` 会自动处理。

- Q：如何确保导出的 beta 图与模板空间一致？
  - A：如果使用 `SimpleITK.Image` 作为输入，导出时用 `CopyInformation` 从某个模板影像复制空间信息后保存。

- Q：分类的 beta_map 返回的是什么？
  - A：对于多分类线性模型，返回 `dict[类别 -> 3D 图]`；二分类或单向量则直接返回 3D 数组。

- Q：`cv_utils` 中的导入报错？
  - A：请按本教程的包导入方式使用（`from pyleison_map import cv_utils`）。若仍失败，检查 `sys.path`，或直接在脚本顶部插入项目绝对路径。

---

## 目录结构简述

- `pyleison_map/lesion_models.py`：核心模型与预处理器、`make_model` 工厂
- `pyleison_map/cv_utils.py`：交叉验证与网格搜索，支持 `est__`/`prep__`/顶层参数设置
- `LESYMAP/`：R 版包与 C++ 实现（与本 Python 教程无直接依赖）

---

## 小结

本教程提供了基于 dTLVC 的病灶-症状建模从数据准备到评估与可视化的完整路径。你可以用线性/树/非线性模型进行回归或分类，利用 `cv_utils` 快速完成 K 折评估与参数搜索，并将系数/重要性投射回 3D 脑空间以解释模型结果。根据任务与数据特点选择合适的指标和超参数，即可在你的队列上开展系统的 LSM 分析。
