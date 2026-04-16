# 给 Codex 的短版任务说明：继续完善 PRIMAL 的结构类 3 通道

## 0. 项目位置
本地项目目录：`D:\Study\RL_AGV\PRIMAL`

你需要修改的主文件是：`mapf_gym.py`。

我会把 PRIMAL2 相关文件内容直接复制给你。**你只需要阅读并参考其语义与实现思路，不需要我提供具体路径，也不要要求路径。**

---

## 1. 本次任务范围
只继续完善 **结构类 3 通道**：

- `blocking_map`
- `delta_x_map`
- `delta_y_map`

**不要改**：

- `pred_1 / pred_2 / pred_3`
- 目标向量 `[dx, dy, mag]`
- `_observe()` 的通道顺序
- `ACNet.py` 输入通道数（当前已经是 10）
- 训练 notebook 接口
- 不要加入 `pathlength_map`
- 不要把整个项目重构成 PRIMAL2 的 `World / Observer` 架构

也就是说，当前 10 通道顺序保持不变：

1. `poss_map`
2. `goal_map`
3. `goals_map`
4. `obs_map`
5. `blocking_map`
6. `delta_x_map`
7. `delta_y_map`
8. `pred_1`
9. `pred_2`
10. `pred_3`

---

## 2. 当前实现存在的问题
当前 `mapf_gym.py` 里已经有 corridor cache 和 10 通道 `_observe()`，但结构类 3 通道仍是简化版，主要问题有：

### 2.1 corridor 检测过于保守
当前 `_is_corridor_cell()` 只识别“非常标准的直走廊内部格”：

- 必须正好有 2 个自由邻居
- 且两个邻居必须严格 opposite
- 另外两侧必须是障碍或越界

这样会漏掉：

- dead-end corridor
- corridor 入口附近
- 与 decision point / stopping point 相关的关键格子

### 2.2 `delta_x_map / delta_y_map` 写在 endpoint 上
当前是把非零值直接写在 corridor endpoint 上。这样表达不够贴近决策语义。

更合理的是：

- 把非零值主要写在 **stopping point / decision point** 上
- 因为 agent 真正需要这些信息的时刻，是“在走廊外准备进入走廊时”

### 2.3 `blocking_map` 过于粗糙
当前逻辑接近：

- 只要某条 corridor 里有其他 agent
- 就把这条 corridor 的 endpoint 标成 1

这没有区分：

- 其他 agent 是否正朝当前入口方向移动
- 是 dead-end 还是双出口 corridor
- 当前观察者是否已经在同一条 corridor 内
- 远端 endpoint 是否被占据
- 哪一个 stopping point 才是真正“当前不该进入”的位置

---

## 3. 期望结果
目标是：**把结构类 3 通道改成更接近 PRIMAL2 语义，但仍保持当前 PRIMAL 项目的代码结构。**

### 3.1 `delta_x_map / delta_y_map`
这两张图表达的是：

- 从当前 stopping point 这一侧进入 corridor 后，另一端相对本端的位移方向

期望语义：

- 每条双出口 corridor 有两个 endpoint：`e0`, `e1`
- 每个 endpoint 对应一个 stopping point
- 若当前 stopping point 对应 `e0`，则：
  - `delta_x = e1.x - e0.x`
  - `delta_y = e1.y - e0.y`
- 若当前 stopping point 对应 `e1`，则反向填写
- 若是 dead-end corridor，则该 stopping point 上：
  - `delta_x = 0`
  - `delta_y = 0`

**重点**：非零值优先写在 **stopping point**，而不是 corridor 内部 endpoint。

### 3.2 `blocking_map`
这张图表达的是：

- **从当前 stopping point 这一侧进入 corridor 是否会被阻塞 / 不建议进入**

期望语义尽量参考 PRIMAL2：

#### dead-end corridor
- 如果 corridor 内已有其他 agent，则该 stopping point 记为 `1`
- 否则 `0`

#### 双出口 corridor
对某个 stopping point：
- 如果 corridor 内存在其他 agent，并且该 agent 的运动趋势会朝当前 stopping point 这一侧退出，则记为 `1`
- 如果 corridor 内靠近当前 stopping point 的位置已被其他 agent 占据，也可记为 `1`
- 如果远端 endpoint 被占据，可优先参考 PRIMAL2 语义记为 `-1`
- 否则 `0`

推荐最终允许取值：
- `1`：当前侧明确阻塞，不应进入
- `0`：当前侧没有明显阻塞
- `-1`：远端 endpoint 被占据

如果保留 `-1` 太麻烦，也可以退化成 `0/1`，但优先尝试保留 `-1/0/1`。

---

## 4. 你需要重点改动的地方
### 必改 1：重做 corridor / endpoint / stopping point 的构建逻辑
不要继续依赖当前“严格直走廊内部格”的 corridor 定义。

请参考 PRIMAL2 的 corridor 语义，至少在当前项目里明确区分：

- obstacle
- free cell outside corridor
- corridor internal cell
- corridor endpoint
- stopping point / decision point

建议维护或新增这些缓存：

- `self.corridor_id_map`
- `self.corridor_type_map`
- `self.corridor_cells`
- `self.corridor_endpoints`
- `self.corridor_stopping_points`
- `self.endpoint_to_corridor`
- `self.stopping_point_to_corridor`
- `self.stopping_point_meta`

### 必改 2：重写 `delta_x_full / delta_y_full` 的生成逻辑
当前是 endpoint sparse map。请改成：

- 非零值主要写在 stopping point 上
- 双出口 corridor 写对端位移
- dead-end 写 0

### 必改 3：重写 `blocking_map` 的生成逻辑
不要再使用“corridor 内只要有人，就把 endpoint 全标 1”的规则。

请改成基于：

- 当前观察者对应的是哪一个 stopping point
- 该 stopping point 对应哪条 corridor
- corridor 内其他 agent 的当前位置
- 其他 agent 的移动趋势 / 下一步方向
- 远端 endpoint 是否被占据

来决定 blocking 值。

### 必改 4：保证 `_observe()` 接口不变
你可以重写其内部依赖的 helper，但 `_observe()` 最终仍应返回当前 10 通道，不改返回格式。

---

## 5. 推荐实现方式
### 5.1 不要照搬 PRIMAL2 的类结构
PRIMAL2 的 `Primal2Observer.py` / `Env_Builder.py` 只作为语义和算法参考。

不要把当前 PRIMAL 项目改造成：
- 新的 `World`
- 新的 `ObservationBuilder`
- 新的 observer 框架

正确做法是：

- 保留当前 `MAPFEnv`
- 在 `mapf_gym.py` 内补齐 corridor cache 与结构图构建逻辑

### 5.2 未来预测 3 通道保持原样
当前做法已经够用：

- 只对可见邻居预测
- 用单智能体 A* cost map
- 生成 3 张未来占用二值图

不要动。

---

## 6. 完成标准
当你完成本次修改时，应满足：

1. `_observe()` 的 10 通道顺序不变
2. `pred_1~pred_3` 行为不变
3. 结构类 3 通道不再是“第一版粗糙实现”
4. `delta_x_map / delta_y_map` 主要在 stopping point 上表达方向信息
5. `blocking_map` 反映“当前这一侧是否不该进入 corridor”，而不是“corridor 里是否有人”
6. 代码风格保持接近当前 `mapf_gym.py`
7. 不破坏现有训练接口

---

## 7. 你输出时请包含这些内容
完成代码修改后，请同时给出：

1. 你修改了哪些函数
2. corridor / endpoint / stopping point 在当前项目中的最终定义
3. `blocking_map` 的最终判定规则
4. `delta_x_map / delta_y_map` 的最终写入位置和含义
5. 建议补充的最小单元测试列表

