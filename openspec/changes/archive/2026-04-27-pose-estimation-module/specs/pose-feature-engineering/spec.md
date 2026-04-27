## ADDED Requirements

### Requirement: 关节角度计算
系统 SHALL 能够基于三个关键点的坐标计算关节角度（弧度和角度制）。系统 SHALL 预定义一组核心关节角度配置，包括左/右肘角、左/右膝角、左/右肩角、左/右髋角。

#### Scenario: 计算单个关节角度
- **WHEN** 提供三个关键点（如肩、肘、腕）的坐标
- **THEN** 返回该关节的角度值（角度制，范围 [0, 180]）

#### Scenario: 批量计算核心关节角度
- **WHEN** 输入一个 PoseFrame
- **THEN** 返回包含所有预定义核心关节角度的字典，键为关节名称（如 "left_elbow"），值为角度值

#### Scenario: 低可见度关键点处理
- **WHEN** 参与计算的关键点中有任一 visibility < 0.5
- **THEN** 该关节角度 SHALL 标记为 NaN，并记录警告日志

### Requirement: 骨骼长度比计算
系统 SHALL 能够计算相邻关键点之间的欧氏距离，并生成骨骼长度比值，用于姿态归一化。

#### Scenario: 计算骨骼段长度
- **WHEN** 输入一个 PoseFrame
- **THEN** 返回预定义骨骼段（如上臂、前臂、大腿、小腿等）的长度值

#### Scenario: 计算归一化骨骼比例
- **WHEN** 输入一个 PoseFrame
- **THEN** 以躯干长度（肩中点到髋中点的距离）为基准，返回各骨骼段的归一化比值

### Requirement: 运动速度计算
系统 SHALL 能够基于相邻帧的关键点位移和时间差计算各关键点的瞬时速度和加速度。

#### Scenario: 计算关键点速度
- **WHEN** 输入连续两个 PoseFrame
- **THEN** 返回 33 个关键点各自的瞬时速度（像素/秒），形状为 (33,)

#### Scenario: 计算序列速度
- **WHEN** 输入一个 PoseSequence（≥2 帧）
- **THEN** 返回形状为 (T-1, 33) 的速度数组

#### Scenario: 单帧序列处理
- **WHEN** 输入的 PoseSequence 仅包含 1 帧
- **THEN** 返回空数组，不抛出异常

### Requirement: 综合特征向量生成
系统 SHALL 能够将关节角度、骨骼比例和运动速度组合为统一的特征向量，作为下游 DTW 对比和深度学习模型的输入。

#### Scenario: 生成单帧特征向量
- **WHEN** 输入一个 PoseFrame
- **THEN** 返回一维特征向量，包含所有核心关节角度和骨骼比例

#### Scenario: 生成序列特征矩阵
- **WHEN** 输入一个 PoseSequence
- **THEN** 返回形状为 (T, D) 的特征矩阵，其中 D 为特征维度
