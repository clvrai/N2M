# Predictor实现说明

本文档说明各个predictor的实际对接情况和使用要求。

## 实现状态总览

| Predictor | 状态 | 需要的外部依赖 | 说明 |
|-----------|------|---------------|------|
| **BlankPredictor** | ✅ 完全实现 | 无 | 无需任何模型，直接返回当前pose |
| **N2MPredictor** | ⚠️ 部分实现 | N2M模块 | 框架完整，需对接N2M模块细节 |
| **MobipiPredictor** | ⚠️ 部分实现 | Mobipi模块、3DGS模型 | 框架完整，需完善渲染流程 |
| **LeLaNPredictor** | ⚠️ 部分实现 | LeLaN模块 | 框架完整，需确认LeLaN接口 |
| **ReachabilityPredictor** | 📝 框架only | IK求解器 | 留给用户实现 |

## 详细说明

### 1. BlankPredictor ✅

**状态**: 完全可用

**实现**: `benchmark/predictor/blank_predictor.py`

**用法**:
```bash
python scripts/run_benchmark.py predictor=blank
```

**说明**: 
- 直接返回当前robot pose，不做任何预测
- 作为baseline评估navigation的必要性
- 无需任何checkpoint或配置

---

### 2. N2MPredictor ⚠️

**状态**: 框架完整，需对接N2M模块

**实现**: `benchmark/predictor/n2m_predictor.py`

**已实现**:
- ✅ 点云生成pipeline (RGB-D → point cloud)
- ✅ GMM采样逻辑
- ✅ 碰撞检测集成
- ✅ 基本inference流程

**需要对接**:
1. **N2M模块加载**:
   ```python
   from n2m.module import N2Mmodule
   ```
   - 需要确认`N2Mmodule`的实际接口
   - 当前假设有`forward_inference(point_cloud)`方法
   - 输出格式：`{'means', 'log_vars', 'weights'}`

2. **相机参数获取**:
   - 需要在运行时传入camera intrinsics/extrinsics
   - 或在env_info中提供

3. **配置文件**:
   ```yaml
   # configs/predictor/n2m.yaml
   checkpoint_path: data/predictor/n2m/n2m_model.ckpt  # 或.json
   ```

**使用前准备**:
```bash
# 1. 训练N2M模型（参考predictor/N2M/README.md）
cd predictor/N2M
python scripts/train.py --config configs/training/config.json

# 2. 放置checkpoint
cp path/to/trained/model.ckpt data/predictor/n2m/

# 3. 运行benchmark
python scripts/run_benchmark.py predictor=n2m
```

---

### 3. MobipiPredictor ⚠️

**状态**: 框架完整，需完善3DGS渲染

**实现**: `benchmark/predictor/mobipi_predictor.py`

**已实现**:
- ✅ Feature encoder加载 (DINO, Policy, DINO Dense)
- ✅ 贝叶斯优化框架
- ✅ BatchSceneModel集成

**需要完善**:
1. **3DGS渲染流程**:
   ```python
   def _render_from_pose(self, pose):
       # 需要实现：
       # 1. SE2 pose → camera extrinsics转换
       # 2. 调用BatchSceneModel.render()
       # 3. 处理多相机场景
   ```

2. **相机配置**:
   - BatchSceneModel需要camera intrinsics
   - 需要相对相机位姿(relative camera poses)

3. **Score function**:
   - 当前使用HybridDistribution
   - 需确认与目标特征的similarity计算方式

**使用前准备**:
```bash
# 1. 构建3DGS场景模型
cd predictor/mobipi/mobipi/scene_model
# 收集多视角图像 → 训练3DGS

# 2. 放置模型
cp -r path/to/scene_models data/predictor/mobipi/

# 3. 运行benchmark
python scripts/run_benchmark.py \
  predictor=mobipi \
  predictor.encoder_type=dino_dense_descriptor
```

---

### 4. LeLaNPredictor ⚠️

**状态**: 框架完整，需确认LeLaN接口

**实现**: `benchmark/predictor/lelan_predictor.py`

**已实现**:
- ✅ 迭代预测框架
- ✅ 控制量→pose delta转换
- ✅ 图像预处理(resize to 224x224)

**需要确认**:
1. **LeLaN模块导入**:
   ```python
   from lelan.nav_model import LeLaNModel  # 需确认实际路径
   ```

2. **Forward接口**:
   ```python
   action = lelan_model.forward(image_224x224, instruction)
   # 输出应有: action.linear.x, action.angular.z
   ```

3. **Checkpoint格式**:
   - 当前假设有`load_from_checkpoint()`方法
   - 需确认实际加载方式

**使用前准备**:
```bash
# 1. 训练LeLaN (参考predictor/lelan)
# 2. 放置checkpoint
cp path/to/lelan.pth data/predictor/lelan/checkpoints/

# 3. 运行benchmark
python scripts/run_benchmark.py \
  predictor=lelan \
  predictor.task_description="navigate to the target object"
```

---

### 5. ReachabilityPredictor 📝

**状态**: 框架only，留给用户实现

**实现**: `benchmark/predictor/reachability_predictor.py`

**需要实现**:
```python
def predict(self, observation, current_pose, env_info):
    # 1. 获取目标物体位置
    target_pos = env_info['position']
    
    # 2. 采样候选pose
    candidate_pose = sample_near_target(target_pos)
    
    # 3. 碰撞检测
    if collision_checker.check_collision(candidate_pose):
        return current_pose, False, {}  # Continue
    
    # 4. IK可达性检查
    if not check_ik_reachable(candidate_pose, target_pos):
        return current_pose, False, {}  # Continue
    
    # 5. 找到合法pose
    return candidate_pose, True, {'success': True}
```

---

## 如何测试Predictor

### 快速测试（不需要实际模型）

```bash
# 1. 测试BlankPredictor（完全可用）
python scripts/run_benchmark.py \
  env=PnPCounterToCab \
  predictor=blank \
  benchmark.num_episodes=5

# 成功标志：能完整运行并生成results.json
```

### 测试实际Predictor（需要模型）

**前提条件**:
- 已训练好对应predictor的模型
- Checkpoint文件已放置在正确位置
- 依赖包已安装

**测试步骤**:
```bash
# 1. 检查checkpoint路径
ls -la data/predictor/n2m/n2m_model.ckpt

# 2. 测试单个episode
python scripts/run_benchmark.py \
  env=PnPCounterToCab \
  predictor=n2m \
  benchmark.num_episodes=1 \
  +debug=true

# 3. 检查日志输出
# 应该看到predictor加载成功，预测运行无报错
```

---

## 常见问题

### Q1: ImportError: No module named 'n2m'

**原因**: N2M模块未安装

**解决**:
```bash
cd predictor/N2M
pip install -e .
```

### Q2: 相机intrinsics未找到

**原因**: 环境创建时未正确获取相机参数

**解决**: 确保`env_utils.py`中的`get_camera_params()`能正确访问环境

### Q3: 3DGS渲染报错

**原因**: BatchSceneModel未正确初始化

**解决**: 检查scene_model目录结构，确保包含nerfstudio需要的所有文件

---

## 下一步工作

根据实际测试结果，需要：

1. **N2M**: 对接实际N2Mmodule接口
2. **Mobipi**: 完善`_render_from_pose()`方法
3. **LeLaN**: 确认`lelan.nav_model`导入路径
4. **All**: 添加更详细的错误处理和日志输出

完成这些对接后，所有predictor将可以实际运行。
