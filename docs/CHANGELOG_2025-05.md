# 人体动作矫正系统 — 系统改造文档

> 本文档整理了一轮迭代中对"人体动作矫正系统"的全部最终改动，
> 用于后续回顾、交接与整理。
> 每个模块按 **问题 → 根因 → 方案 → 代码变更 → 验证结果** 的顺序描述。

---

## 目录

- [一、概览](#一概览)
- [二、问题 1：实时模式骨骼抖动 / 丢失 / 跟不上运动](#二问题-1实时模式骨骼抖动--丢失--跟不上运动)
- [三、问题 2：拍摄角度 / 身材差异干扰对比](#三问题-2拍摄角度--身材差异干扰对比)
- [四、问题 3：模板权威性 — 自动剔除模板不可见关节](#四问题-3模板权威性--自动剔除模板不可见关节)
- [五、问题 4：模板骨骼演示视频](#五问题-4模板骨骼演示视频)
- [六、问题 5a：动作不符合自动判定](#六问题-5a动作不符合自动判定)
- [七、问题 5b：帧对比查看器](#七问题-5b帧对比查看器)
- [八、问题 6：完成度评分（核心改动）](#八问题-6完成度评分核心改动)
- [九、附录：配置项清单](#九附录配置项清单)
- [十、附录：主要文件改动清单](#十附录主要文件改动清单)

---

## 一、概览

### 解决的核心问题
1. **实时模式**骨骼抖动、偶发丢失、跟不上动作
2. **拍摄角度 / 身材差异**对对比结果影响过大
3. **模板权威性**：模板只录了上/下半身时，对比和建议应只聚焦可见部位
4. **模板骨骼演示视频**：录入模板后希望在模板管理里"点击即播放"
5. **矫正报告**：偏差过大时应拒绝给建议；偏差不大时提供逐帧骨骼对比
6. **评分不公**：用户只做了模板的一部分（如模板含多个动作、只完成其中一个）仍能获得高分

### 高层架构不变
- 姿态估计：MediaPipe Pose Landmarker
- 动作对比：动态时间规整（DTW）
- 矫正建议：规则引擎（`CorrectionRuleEngine`）+ 关节偏差分析
- UI：Gradio 多 Tab 应用

---

## 二、问题 1：实时模式骨骼抖动 / 丢失 / 跟不上运动

### 根因
- 原实时链路使用 `RunningMode.IMAGE`，**每帧独立检测**，缺少时序追踪
- 早期尝试过的 EMA 平滑（`alpha=0.55, hold=6`）过度保守，导致跟不上动作

### 最终方案
1. 新增 `StreamingPoseEstimator`，使用 **`RunningMode.LIVE_STREAM`** + MediaPipe 自带跨帧追踪
2. 保留 `PoseSmoother` 作为 MediaPipe 之上的轻量微平滑，参数调整为：
   - `alpha = 0.80`（跟手更紧）
   - `min_alpha_floor = 0.55`
   - `max_hold_frames = 3`（≈ 100ms @ 30fps，短暂遮挡即可恢复）
3. 摄像头和实时反馈各持有独立 `StreamingPoseEstimator` 实例，避免时间戳冲突

### 关键代码
- **`src/pose_estimation/estimator.py`**
  - 新增类 `StreamingPoseEstimator`
    - 通过 `PoseLandmarker.create_from_options(running_mode=LIVE_STREAM, result_callback=...)` 启动
    - `process(image_bgr, timestamp_ms)` 同步等待回调（最多 80ms），强制时间戳单调递增
  - `PoseSmoother` 参数从 `0.55/6` 调为 `0.80/3`

- **`src/app/pipeline.py`**
  ```python
  self._cam_streamer: Optional[StreamingPoseEstimator] = None
  self._rt_streamer: Optional[StreamingPoseEstimator] = None
  self._cam_smoother = PoseSmoother(alpha=0.80, min_alpha_floor=0.55,
                                     max_hold_frames=3)
  self._rt_smoother  = PoseSmoother(alpha=0.80, min_alpha_floor=0.55,
                                     max_hold_frames=3)
  ```
  - `process_camera_frame` / `process_realtime_frame` 改用 `StreamingPoseEstimator`
  - 新增 `reset_camera_smoothers()` 会同时重置 streamer 和 smoother
  - `close()` 中释放 `_cam_streamer` 和 `_rt_streamer`

### 验证
用户实测：骨骼稳定、跟手、短暂遮挡能自动恢复、不再有原先的高频抖动。

---

## 三、问题 2：拍摄角度 / 身材差异干扰对比

### 根因
- 原对比直接使用原始归一化坐标 `(lm.x, lm.y, lm.z)`，侧身/前视差异直接进入距离函数
- 身高、骨架尺寸不同 → 完全相同的动作距离值悬殊

### 最终方案
坐标归一化三件套（**只用于数值对比，不用于渲染**）：
1. **髋中点平移归一化** — 消除人在画面中的位置差异
2. **躯干长度缩放** — 消除身高 / 肌肉差异
3. **肩轴 2D 旋转对齐** — 左右肩连线对齐到 +x 方向，消除侧身/前视/朝向差异

相似度本身基于**纯角度特征**（`sequence_to_feature_matrix` → 9 维角度向量），天然是视角不变量。

### 关键代码
- **`src/action_comparison/distance_metrics.py`**
  - 新增 `_rotation_from_shoulder_axis(lm)` 从肩膀向量计算 2×2 旋转矩阵
  - 新增 `sequence_to_landmark_matrix_masked(sequence, joint_indices, align_shoulder=True)`
    仅在**数值对比层**使用

- **`src/action_comparison/deviation_analyzer.py`**
  - 将内部坐标矩阵换成 `sequence_to_landmark_matrix_masked(..., align_shoulder=True)`

### ⚠️ 重要原则
`align_shoulder` **绝对不可以**用于渲染路径，否则会把骨骼扭曲变形（历史教训）。渲染一律使用原始 landmarks + 平移缩放（见问题 7 "帧对比查看器"）。

---

## 四、问题 3：模板权威性 — 自动剔除模板不可见关节

### 根因
如果模板视频只录了上半身（下半身出画），下半身关节在模板里 `visibility ≈ 0`，但仍会进入距离计算，造成噪声。

### 最终方案
从模板自动检测"**有效关节集**"：
- 关节在模板中有 ≥ 60% 的帧 `visibility ≥ 0.3` 时视为"模板关注的关节"
- 其他关节从对比距离、偏差分析、矫正建议中**全部剔除**
- 被剔除的关节名在报告中列出："【模板关注范围】已自动剔除模板中不可见的关节：左膝、左踝、..."

### 关键代码
- **`src/action_comparison/distance_metrics.py`**
  ```python
  def compute_valid_joints(
      sequence,
      visibility_threshold: float = 0.3,
      presence_ratio_threshold: float = 0.6,
  ) -> list:
      """返回模板中"有效关节"的 MediaPipe 索引列表"""
  ```
- **`src/action_comparison/deviation_analyzer.py`**
  - `analyze()` 开头：`valid_joints = compute_valid_joints(t_seq)`
  - 仅对 `valid_joints` 做偏差计算
  - `DeviationReport` 新增字段 `valid_joint_indices` 和 `excluded_joint_names`

- **`src/correction/data_types.py`**
  - `CorrectionReport` 新增 `excluded_joints: List[str]`
  - `to_text()` 中打印："【模板关注范围】已自动剔除模板中不可见的关节：..."

- **`src/correction/feedback.py`**
  - `generate()` 把 `DeviationReport.excluded_joint_names` 透传到 `CorrectionReport`

### 验证
- `throw1 vs throw`：Top 偏差正确聚焦 `right_wrist / right_elbow`
- `throw1 vs push`：`push` 模板中"左肘"被自动识别为不可见并剔除

---

## 五、问题 4：模板骨骼演示视频

### 根因
用户希望：
- 录入模板后直接在模板管理里预览骨骼动画
- 切换下拉即播放，不要"点生成"的等待
- 风格跟视频分析右半的"标准模板"骨骼图一致
- 能正常播放（不要 mp4v 浏览器解不开）

### 最终方案
1. **录入模板时自动生成** `.demo.mp4`，存在模板 JSON 旁边
2. **切换下拉即加载**，`gr.Video(autoplay=True)`
3. **复用**视频分析中的渲染函数 `_draw_skeleton_on_region`，保证视觉一致
4. **编码器按顺序尝试** `avc1 → H264 → mp4v`（浏览器友好）
5. **居中 + 等比缩放**：渲染前把整段序列的 landmarks 平移缩放到画布中心，防止骨骼挤在角落（关键！只做平移缩放，不做旋转，**不改变骨骼形态**）

### 存储约定
```
data/templates/
  └── <action>/
      ├── <template>.json        (PoseSequence JSON)
      └── <template>.demo.mp4    (骨骼演示视频，删除模板时一并清理)
```

### 关键代码
- **新增 `src/correction/template_video.py`**
  - `_compute_sequence_fit_box(sequence)` → `(cx, cy, half_w, half_h, scale_ref)`
  - `_fit_frame_to_canvas(frame, fit_box)` 线性平移+等比缩放，不扭曲
  - `render_template_demo_video(sequence, output_path, title)` 生成 MP4
  - `render_pair_frame(user_frame, template_frame, user_fit_box, template_fit_box, ...)` 用于帧对比
  - `_CODEC_CANDIDATES = ("avc1", "H264", "mp4v")`

- **`src/data/template_library.py`**
  - 新增 `@property root`
  - `delete_template()` 和 `delete_action()` 同步清理 `.demo.mp4`

- **`src/app/pipeline.py`**
  - `record_template()` 成功后调用 `self._build_template_demo()`
  - 新增 `get_template_demo_path(action, template_name, build_if_missing=True)`
    - 文件存在 → 直接返回路径（秒开）
    - 文件不存在且 `build_if_missing=True` → 渲染一次后返回

- **`src/app/gradio_ui.py`**
  - 模板管理 Tab 底部新增"🦴 模板骨骼演示视频"
  - 下拉切换自动回调 `on_tpl_preview_template_change` 切视频源
  - 无"生成"按钮，无"强制重新生成"选项（已移除）

### 注意事项
- OpenCV `putText` 不支持中文字符，所有叠加文字一律使用英文（例如 `Your Action`、`Template`）

---

## 六、问题 5a：动作不符合自动判定

### 根因
偏差极大时仍给"具体矫正建议"会误导用户（例如跑步 vs 深蹲，强行给"请弯曲左膝 xx°" 毫无意义）。

### 最终方案
两种情况自动判定"不符合"：
1. **姿势质量过低** — `raw_similarity < mismatch_similarity_threshold`（默认 0.5）
2. **完成度过低** — `coverage_factor < 0.1`

触发后：
- `report.mismatch = True`
- **清空** `report.corrections`
- 覆盖整体评语，明确告诉用户"不建议基于此给出矫正提示"
- 报告顶部显示 ⛔"动作不符合"区块 + 具体判定原因

### 关键代码
- **`src/correction/data_types.py`**
  ```python
  mismatch: bool = False
  mismatch_reason: str = ""
  ```
  `to_text()` 中在整体评语之后插入 mismatch 区块，直接 return，**不输出矫正建议**。

- **`src/correction/pipeline.py`** — `analyze()` 末尾
  ```python
  raw_sim = report.raw_similarity
  too_incomplete = report.coverage_factor < 0.1
  too_dissimilar = raw_sim < self._mismatch_threshold

  if too_incomplete or too_dissimilar:
      report.mismatch = True
      if too_incomplete:
          report.mismatch_reason = (
              f"动作完成度仅 {report.template_coverage:.1%}，"
              f"低于最低阈值 {self._comparator._coverage_hard_floor:.0%} — "
              f"用户动作只覆盖了模板的一小段，无法进行有效矫正"
          )
      else:
          report.mismatch_reason = (
              f"相似度 {raw_sim:.2%} 低于不符合阈值 "
              f"{self._mismatch_threshold:.2%}"
          )
      report.corrections = []
  ```

### 配置项
```yaml
correction:
  mismatch_similarity_threshold: 0.5   # raw_similarity 低于此值判定不符合
```

---

## 七、问题 5b：帧对比查看器

### 根因 & 多次迭代教训
经过两次返工，最终方案是：
- ❌ **不能**使用归一化/旋转坐标来渲染（会扭曲骨骼）
- ❌ **不能**使用 `processed_query` / `processed_template`（那是被重采样过的插值序列，会失真）
- ✅ 使用**原始未预处理**的 `user_sequence` 和 `template_sequence`
- ✅ 使用**对比视频同款** DTW 帧映射 `_build_direct_dtw_map`（保证"滑条拖到用户帧 X 时，右侧模板帧就是它应该同步到的那一帧"）
- ✅ 渲染前做**居中+等比缩放**（平移+等比缩放，不变形），左右两侧使用**统一的 scale**让骨骼大小一致

### 关键代码
- **`src/correction/pipeline.py`**
  - `analyze()` 末尾：`report._user_sequence = user_sequence` 保留原始用户序列

- **`src/app/pipeline.py`**
  - 新增 `prepare_frame_viewer(report)` → 返回 `viewer_state` dict：
    ```python
    {
      "user_seq": 原始用户 PoseSequence,
      "template_seq": 原始模板 PoseSequence,
      "dtw_map": {user_frame_idx: tpl_frame_idx},
      "match_start": int,
      "match_end": int,
      "user_fit_box": (cx, cy, hw, hh, scale),
      "template_fit_box": (...),  # scale 统一取两者较大值
      "joint_deviations": ...,
      "total_user_frames": int,
    }
    ```
  - 新增 `render_viewer_frame(viewer_state, user_frame_idx, width, height)` 
    按用户帧号返回并排骨骼图 + 关节偏差表 + 逐帧建议

- **`src/app/gradio_ui.py`**
  - 视频分析 Tab 底部新增"🔍 帧对比查看器"：
    - `frame_viewer_slider` 拖条（拖的是用户帧号，1-based）
    - `frame_viewer_image`（并排骨骼 960×640）
    - `frame_viewer_info` 显示"用户帧 X/N ↔ 模板帧 Y/M · 阶段：同步"
    - `frame_viewer_table` 关节偏差 Markdown 表（🔴🟡🟢 + 中英文名 + 数值）
    - `frame_viewer_advice` 针对该帧的三条具体建议
  - 分析完成后 `on_frame_viewer_init` 一次性构建 DTW 帧映射并渲染首帧
  - 滑条变化调用 `on_frame_viewer_render`

### 注意事项
- 滑条标签格式：`用户帧（共 {N} 帧） — 拖动查看动作对比`
- 在"准备段"（用户帧 < match_start）显示模板第 0 帧
- 在"收尾段"（用户帧 > match_end）显示模板最后一帧
- 骨骼中间画分隔线、顶部标签 `Your Action` / `Template`

---

## 八、问题 6：完成度评分（核心改动）

### 根因
- `dtw_algorithms.compute_dtw` 中有一个 **"角色交换"** 分支：当模板更长时，为了"让两个序列都能用子序列 DTW"，会把模板当作 query、用户当作 template 交给 `subsequence_dtw` → **模板会被截断**，只找出一段与用户最像的模板子序列
- 结果：用户只完成模板的一小段（例如模板有 5 个动作、只做了 1 个），**仍能得到高相似度和高分**
- 实时模式和视频分析共享同一个 `ActionComparator`，问题同样存在

### 最终方案
**"以模板为权威"的双维评分**：

1. **废除角色交换**（`compute_dtw`）
   - 始终让模板作为完整第二维，模板的每一帧必须被匹配
   - 用户 ≥ 模板 → 子序列 DTW（允许用户开端/收尾弹性）
   - 用户 < 模板 → 退化为经典 DTW（两端强制对齐，模板没做完的部分会累积大距离）

2. **真实完成度计算**
   - `coverage = cropped_q / cropped_t`（去除前后静止段后、预处理前的帧数比）
   - **不**使用"DTW 路径覆盖的模板唯一帧数"（那在 classic DTW 下恒为 1.0，失去区分力）

3. **双阀限 + 二次曲线 f(coverage)**
   ```
   coverage ≤ 30%  → factor = 0     （评分归零，触发 mismatch）
   coverage ≥ 90%  → factor = 1     （不降分）
   中间             → factor = t²   (t = (cov - 30%) / (90% - 30%))
   ```
   二次曲线让"接近全部完成"的用户获得明显奖励。

4. **双维度评分**
   - **姿势质量** = `raw_similarity`（仅看动作做得对不对）
   - **完成度** = `coverage`（用户动作时长覆盖模板比例）
   - **最终分数** = `raw_similarity × f(coverage)`
   - 报告中三个数字分开展示

### 关键代码

#### `src/action_comparison/dtw_algorithms.py`
```python
# 废除角色交换
if use_subsequence:
    distance, path, cost, _, _ = subsequence_dtw(
        q_feat, t_feat, dist_func, window_size
    )
    return distance, path, cost
```

#### `src/action_comparison/comparison.py`
```python
@dataclass
class ComparisonResult:
    # 原始字段 + 新增
    raw_similarity: float = 0.0           # 未应用覆盖度前的相似度
    template_coverage: float = 1.0        # 用户动作覆盖模板的比例
    coverage_factor: float = 1.0          # 完成度惩罚因子

class ActionComparator:
    def __init__(self, ..., coverage_min_full=None, coverage_hard_floor=None):
        # 从 configs/default_config.yaml 读取默认值

    def compare(self, query, template, template_name=""):
        # ...原有预处理 + DTW...

        # 计算 raw_similarity (高斯核映射)
        raw_similarity = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))

        # 计算完成度（基于裁剪后帧数比）
        if cropped_t > 0 and path:
            template_coverage = min(cropped_q / float(cropped_t), 1.0)
        else:
            template_coverage = 0.0

        # 二次曲线 f(coverage)
        if template_coverage <= hard_floor:
            coverage_factor = 0.0
        elif template_coverage >= min_full:
            coverage_factor = 1.0
        else:
            t = (template_coverage - hard_floor) / (min_full - hard_floor)
            coverage_factor = t * t

        similarity = raw_similarity * coverage_factor

        return ComparisonResult(..., 
            raw_similarity=raw_similarity,
            template_coverage=template_coverage,
            coverage_factor=coverage_factor,
        )
```

#### `src/correction/data_types.py`
```python
@dataclass
class CorrectionReport:
    # ...
    raw_similarity: float = 0.0
    template_coverage: float = 1.0
    coverage_factor: float = 1.0

    def to_text(self):
        lines = [
            ...
            f"【质量评分】{self.quality_score:.1f} / 100",
            "",
            "【评分维度】",
            f"  ├─ 姿势质量 {self.raw_similarity:.1%}  （仅看动作做得对不对）",
            f"  ├─ 完成度   {self.template_coverage:.1%}  （用户动作覆盖模板比例）",
            f"  └─ 折算后相似度 {self.similarity:.1%}  （= 姿势质量 × 完成度函数）",
            ...
        ]
```

#### `src/app/data_types.py`
```python
class AnalysisResult:
    def summary(self):
        # 同时展示 姿势质量 + 完成度 + 折算相似度
```

### 附带修复
**`src/correction/angle_utils.py`** — `compare_angles()` 老 bug
- `path` 索引来自预处理（重采样后）序列，但函数在原始序列上取帧 → `IndexError`
- 已修复为 `ii = min(i, n_user - 1)`, `jj = min(j, n_tmpl - 1)`，空数组时用 `nanmean`

### 验证
| 情景 | 姿势质量 | 完成度 | 折算因子 | 最终分 | mismatch |
|---|---|---|---|---|---|
| 完整 throw1 vs throw | 83.5% | 98.3% | 1.000 | **87.9** | ❌ |
| throw1 前 50% | 71.5% | 44.8% | 0.061 | 2.9 | ✅ |
| throw1 前 30% | 75.2% | 34.5% | 0.006 | 0.3 | ✅ |
| throw1 前 80% | 81.5% | 58.6% | 0.228 | 12.4 | ❌ |
| throw1 前 90% | 82.7% | 77.6% | 0.629 | 47.5 | ❌ |
| throw1 vs push (不同动作) | 64.4% | 100% | 1.000 | 62.9 | ❌ |

✅ 正确区分了"做对了但没做完"(前 50%~90%) 和"完整但做错了"(throw vs push)

---

## 九、附录：配置项清单

`configs/default_config.yaml` 中 `correction` 段新增项：

```yaml
correction:
  quality_excellent: 90                  # 优秀评分线
  quality_good: 70                       # 良好评分线
  quality_fair: 50                       # 需改进评分线
  max_suggestions: 10                    # 最大建议条数

  # === 动作不符合判定 ===
  mismatch_similarity_threshold: 0.5     # raw_similarity 低于此值判定"不符合"

  # === 完成度评分（核心）===
  # 完成度 = cropped_q / cropped_t（裁剪后用户帧数 / 裁剪后模板帧数）
  # factor 函数：
  #   coverage ≤ hard_floor → 0（评分归零）
  #   coverage ≥ min_full   → 1（不降分）
  #   中间                  → t² 二次曲线插值
  # 最终评分 = raw_similarity × factor
  coverage_min_full: 0.90                # 达到 90% 才视作"完整完成"
  coverage_hard_floor: 0.30              # 低于 30% 直接归零
```

调优建议：
- 想更宽松 → `coverage_min_full=0.80, coverage_hard_floor=0.25`
- 想更严格 → `coverage_min_full=0.95, coverage_hard_floor=0.40`
- `mismatch_similarity_threshold` 升高会让更多"看着不像"的动作触发 mismatch

---

## 十、附录：主要文件改动清单

### 新增文件

| 文件 | 作用 |
|---|---|
| `src/correction/template_video.py` | 模板骨骼演示视频渲染 + 并排帧对比渲染器 |

### 修改文件（按模块分组）

#### 姿态估计
- `src/pose_estimation/estimator.py`
  - 新增 `StreamingPoseEstimator`（LIVE_STREAM 模式 + 时序追踪）
  - `PoseSmoother` 参数调整

#### 动作对比
- `src/action_comparison/dtw_algorithms.py`
  - `compute_dtw` 废除"角色交换"分支

- `src/action_comparison/comparison.py`
  - `ComparisonResult` 新增 `raw_similarity / template_coverage / coverage_factor`
  - `ActionComparator.__init__` 新增覆盖度参数，从配置读取默认值
  - `ActionComparator.compare` 计算完成度 + 双维评分 + 二次曲线 f(coverage)

- `src/action_comparison/distance_metrics.py`
  - 新增 `compute_valid_joints`（模板有效关节集检测）
  - 新增 `_rotation_from_shoulder_axis`
  - 新增 `sequence_to_landmark_matrix_masked`

- `src/action_comparison/deviation_analyzer.py`
  - 使用 `compute_valid_joints` 确定对比关节范围
  - 使用 `sequence_to_landmark_matrix_masked(align_shoulder=True)`
  - `DeviationReport` 新增 `valid_joint_indices / excluded_joint_names`

#### 矫正反馈
- `src/correction/data_types.py`
  - `CorrectionReport` 新增：
    - `frame_details`（逐帧详情）
    - `excluded_joints`（模板剔除的关节）
    - `mismatch / mismatch_reason`（动作不符合标记）
    - `raw_similarity / template_coverage / coverage_factor`（评分三维）
  - `to_text()` 重写，分层展示评分维度 + mismatch 区块

- `src/correction/pipeline.py`
  - `analyze()` 末尾透传覆盖度字段、保存 `_user_sequence`、完成度/相似度过低触发 mismatch
  - 构造函数新增 `mismatch_similarity_threshold` 参数

- `src/correction/feedback.py`
  - `generate()` 透传 `excluded_joints`

- `src/correction/angle_utils.py`
  - `compare_angles` 修复 path 索引 IndexError + 空数组 `nanmean`

#### 数据层
- `src/data/template_library.py`
  - 新增 `@property root`
  - `delete_template / delete_action` 连带清理 `.demo.mp4`

#### 应用层
- `src/app/pipeline.py`
  - 引入 `StreamingPoseEstimator` + 两个独立实例
  - `record_template` 成功后自动调用 `_build_template_demo`
  - 新增 `_template_demo_path / _build_template_demo / get_template_demo_path`
  - 新增 `prepare_frame_viewer / render_viewer_frame`（帧对比查看器）
  - `close()` 释放 streamer

- `src/app/data_types.py`
  - `AnalysisResult.summary()` 同时展示姿势质量 + 完成度 + 折算相似度

- `src/app/gradio_ui.py`
  - 模板管理 Tab：新增"🦴 模板骨骼演示视频"（切换即播放，`autoplay=True`）
  - 视频分析 Tab：新增"🔍 帧对比查看器"
  - 删除模板 / 删除动作按钮级联刷新所有下拉框
  - 分析完成后的 `.then(on_frame_viewer_init)` 自动初始化帧查看器

#### 配置
- `configs/default_config.yaml`
  - `correction` 段新增 `mismatch_similarity_threshold / coverage_min_full / coverage_hard_floor`

---

## 十一、特别提醒（维护参考）

1. **渲染 vs 数值对比的坐标绝对分离**
   - 数值对比可用归一化、旋转、缩放（`sequence_to_landmark_matrix_masked`, `align_shoulder=True`）
   - 渲染（演示视频、帧对比查看器）**只能平移+等比缩放**，严禁旋转/归一化，否则骨骼会扭曲变形

2. **OpenCV putText 不支持中文**
   - 所有叠加到图像上的标签、水印一律英文（`Your Action`、`Template`、`frame X/Y`）
   - 中文只出现在 Markdown / 文本报告中

3. **DTW 预处理与原始序列的索引不可混用**
   - `result.path` 的索引对应 `processed_query / processed_template`（重采样后）
   - `user_sequence / best_template` 才是原始序列
   - 使用 `path` 查原始序列时必须夹紧或通过 `_build_direct_dtw_map` 构建映射

4. **模板演示视频与模板生命周期绑定**
   - 录入 → 自动生成
   - 删除模板/删除动作 → 演示视频同步清理
   - `get_template_demo_path(build_if_missing=True)` 兜底：老模板首次点击会生成一次

5. **完成度依赖 `cropped_q / cropped_t`**
   - 这是 `extract_action_segment` 后、`preprocess_pipeline` 重采样前的帧数
   - 如果修改预处理流程，确保这两个值仍能真实反映"用户动作时长 vs 模板动作时长"

---

*文档生成于 2025-05-05*
