可以，下面这版就是更适合直接贴给 Codex 的任务指令。

````markdown
你需要帮助我修改 PRIMAL 项目的 `mapf_gym.py`，目标是把当前 `_observe(agent_id)` 的局部观测从 4 通道扩展到 10 通道，风格参考 PRIMAL2，但第一版不加入 pathlength map。

## 一、当前项目背景

当前 `mapf_gym.py` 中：

- `_observe(agent_id)` 现在返回 4 张局部图：
  1. `poss_map`
  2. `goal_map`
  3. `goals_map`
  4. `obs_map`
- 同时返回目标向量 `[dx, dy, mag]`
- 环境中已经有 `getAstarCosts(start, goal)`，可以返回单智能体 A* 的全局 cost map
- 目前 `MAPFEnv.__init__()` 中已经预留了以下成员变量：
  - `self.static_obs_full`
  - `self.corridor_id_map`
  - `self.delta_x_full`
  - `self.delta_y_full`
  - `self.corridor_cells`
  - `self.corridor_endpoints`
  - `self.corridor_stopping_points`
  - `self.endpoint_to_corridor`
- 目前 `_setWorld()` 的三个分支在创建完 `self.world = State(...)` 之后，都会执行：
  - `self.static_obs_full = (self.world.state == -1).astype(np.float32)`
  - `self._compute_corridor_cache()`

也就是说，cache 初始化入口已经搭好了，现在要实现这些 helper methods，并最终改写 `_observe()`。

---

## 二、最终目标

把 `_observe(agent_id)` 的返回通道顺序改为：

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

目标向量 `[dx, dy, mag]` 保持不变。

也就是说，最终返回格式应为：

```python
([poss_map, goal_map, goals_map, obs_map,
  blocking_map, delta_x_map, delta_y_map,
  pred_1, pred_2, pred_3],
 [dx, dy, mag])
````

注意：第一版不要加入 pathlength map。

---

## 三、corridor 的定义

这里采用第一版可落地的工程定义。

### 1. corridor cell

对一个格子 `(x, y)`：

- 必须是自由格，不是障碍
    
- 只考虑 4 邻域：上、下、左、右
    
- 若它的自由邻居数 `<= 2`，则把它视为 corridor cell
    

### 2. corridor

把所有 corridor cell 按 4 连通做连通分量分解，每个连通分量视为一条 corridor。

### 3. endpoint

对某条 corridor 中的一个格子，如果它存在至少一个 4 邻居满足：

- 该邻居是自由格
    
- 但该邻居不属于这条 corridor
    

则该格子视为 endpoint 候选。

一条 corridor 通常有 1 或 2 个 endpoint：

- 2 个 endpoint：普通 corridor
    
- 1 个 endpoint：dead-end corridor
    

### 4. stopping point

第一版只需要把它作为元信息缓存即可，不要求复杂逻辑。  
可以先简单定义为：

- endpoint 本身
    
- 或 endpoint 外面相邻的自由非-corridor 格子
    

---

## 四、需要实现/维护的成员变量

请确保下面这些成员变量被正确维护：

### 1. `self.static_obs_full`

- 类型：`np.ndarray`
    
- shape：`(H, W)`
    
- dtype：`np.float32`
    
- 含义：全局静态障碍图
    
- 取值：障碍为 `1.0`，非障碍为 `0.0`
    

### 2. `self.corridor_id_map`

- 类型：`np.ndarray`
    
- shape：`(H, W)`
    
- dtype：整数类型
    
- 含义：每个格子所属 corridor 的 id
    
- 取值：
    
    - `-1` 表示不属于任何 corridor
        
    - `0,1,2,...` 表示 corridor id
        

### 3. `self.delta_x_full`

- 类型：`np.ndarray`
    
- shape：`(H, W)`
    
- dtype：`np.float32`
    
- 含义：全局稀疏 delta-x 图
    
- 规则：
    
    - 只在 endpoint 位置非零
        
    - 若 corridor 两端点为 `e1=(x1,y1), e2=(x2,y2)`：
        
        - `delta_x_full[e1] = x2 - x1`
            
        - `delta_x_full[e2] = x1 - x2`
            

### 4. `self.delta_y_full`

- 类型：`np.ndarray`
    
- shape：`(H, W)`
    
- dtype：`np.float32`
    
- 含义：全局稀疏 delta-y 图
    
- 规则：
    
    - `delta_y_full[e1] = y2 - y1`
        
    - `delta_y_full[e2] = y1 - y2`
        

### 5. `self.corridor_cells`

- 类型：`dict[int, list[tuple[int, int]]]`
    
- 含义：每条 corridor 的所有格子
    

### 6. `self.corridor_endpoints`

- 类型：`dict[int, list[tuple[int, int]]]`
    
- 含义：每条 corridor 的 endpoint 列表
    

### 7. `self.corridor_stopping_points`

- 类型：`dict[int, list[tuple[int, int]]]`
    
- 含义：每条 corridor 的 stopping point 列表
    

### 8. `self.endpoint_to_corridor`

- 类型：`dict[tuple[int, int], int]`
    
- 含义：endpoint 坐标到 corridor_id 的映射
    

---

## 五、需要实现的 helper methods

请实现以下函数，并尽量保持代码风格与现有 `mapf_gym.py` 一致。

---

### 1. `_compute_corridor_cache(self)`

#### 作用

预处理当前静态地图上的 corridor 信息，并更新所有 corridor 相关缓存。

#### 要求

- 进入函数后先清空旧缓存，避免残留旧地图数据
    
- 重新计算并更新：
    
    - `self.corridor_id_map`
        
    - `self.delta_x_full`
        
    - `self.delta_y_full`
        
    - `self.corridor_cells`
        
    - `self.corridor_endpoints`
        
    - `self.corridor_stopping_points`
        
    - `self.endpoint_to_corridor`
        

#### 推荐流程

1. 清空旧缓存
    
2. 找到所有 corridor components
    
3. 为每个 component 分配 corridor id
    
4. 识别 endpoint
    
5. 填充 `corridor_id_map`
    
6. 构建 `delta_x_full` 和 `delta_y_full`
    

---

### 2. `_is_free_cell(self, x, y)`

#### 作用

判断某个格子是否是自由格

#### 规则

返回 True 当且仅当：

- `(x, y)` 在地图范围内
    
- `self.world.state[x, y] != -1`
    

注意：  
这里判断的是静态自由格，不把其他 agent 当障碍。

---

### 3. `_get_free_neighbors(self, x, y)`

#### 作用

返回 `(x, y)` 的 4 邻域中所有自由格坐标

#### 输出

`list[tuple[int, int]]`

---

### 4. `_is_corridor_cell(self, x, y)`

#### 作用

判断一个格子是否属于 corridor

#### 判定规则

- 若不是自由格，返回 False
    
- 统计 4 邻域自由邻居数量
    
- 若自由邻居数 `<= 2`，返回 True
    
- 否则 False
    

---

### 5. `_find_corridor_components(self)`

#### 作用

找出整张地图中所有 corridor 的 4 连通分量

#### 输出

`list[list[tuple[int, int]]]`

#### 要求

- 遍历整图
    
- 对未访问的 corridor cell 做 BFS 或 DFS
    
- 每个连通分量作为一条 corridor 返回
    

---

### 6. `_find_corridor_endpoints(self, cells)`

#### 输入

- `cells: list[tuple[int, int]]`
    

#### 作用

给定一条 corridor 的格子集合，找出它的 endpoints

#### 输出

`list[tuple[int, int]]`

#### 判定方式

对 corridor 中每个格子：

- 如果它存在一个 4 邻居，该邻居是自由格，但不属于该 corridor，则它是 endpoint 候选
    

#### 备注

- 一般返回 1 或 2 个点
    
- 第一版不需要过度处理复杂分叉情况，简单实现即可
    

---

### 7. `_build_delta_maps(self)`

#### 作用

根据 `self.corridor_endpoints` 生成：

- `self.delta_x_full`
    
- `self.delta_y_full`
    

#### 规则

- 初始化为全 0
    
- 对每条 corridor：
    
    - 如果有两个 endpoint，则互相填坐标差
        
    - 如果只有一个 endpoint，则该点 delta 保持 0
        

---

### 8. `_extract_fov(self, full_map, top_left, fill_value=0.0)`

#### 输入

- `full_map: np.ndarray`
    
- `top_left: tuple[int, int]`
    
- `fill_value: float`
    

#### 作用

从全局图裁剪出以当前 agent 为中心的局部 FOV 图

#### 输出

- shape 为 `(self.observation_size, self.observation_size)` 的二维数组
    

#### 规则

- 越界区域用 `fill_value` 填充
    
- 非越界区域从 `full_map` 拷贝
    

---

### 9. `_predict_agent_future_positions(self, other_agent_id, horizon=3)`

#### 输入

- `other_agent_id: int`
    
- `horizon: int = 3`
    

#### 作用

用 A* cost map 预测某个邻居未来几步的位置

#### 输出

- `list[tuple[int, int]]`
    
- 长度为 `horizon`
    
- 每个元素表示该邻居未来一步的位置
    

#### 实现要求

使用已有的 `getAstarCosts(start, goal)`：

1. 获取邻居当前坐标 `pos`
    
2. 获取其目标 `goal`
    
3. `costs = self.getAstarCosts(pos, goal)`
    
4. 从 `pos` 开始，连续做 `horizon` 次：
    
    - 看当前位置的 4 邻居
        
    - 选择 `costs` 更小的最佳邻居
        
    - 若没有更优邻居，则停在原地
        
5. 返回每一步预测位置列表
    

#### 注意

- 只考虑静态障碍
    
- 忽略其他 agent
    
- 第一版这样就可以，不需要真正回溯完整 A* 路径
    

---

### 10. `_build_prediction_maps(self, agent_id, visible_agents, top_left, horizon=3)`

#### 输入

- `agent_id: int`
    
- `visible_agents: list[int]`
    
- `top_left: tuple[int, int]`
    
- `horizon: int = 3`
    

#### 作用

构造当前 agent 的三张未来预测图：

- `pred_1`
    
- `pred_2`
    
- `pred_3`
    

#### 输出

`tuple(pred_1, pred_2, pred_3)`

每张图都应为：

- shape = `(self.observation_size, self.observation_size)`
    
- dtype = `np.float32`
    

#### 构造规则

对每个可见邻居：

1. 调 `_predict_agent_future_positions(other_agent_id, horizon)`
    
2. 将第 1、2、3 步的位置分别投影到三张图
    
3. 若落在 FOV 内，对应格置 1
    

#### 第一版要求

- 使用二值图即可
    
- 若多个邻居预测到同一格，仍然只置 1
    

---

### 11. `_build_blocking_map(self, agent_id, top_left, visible_agents)`

#### 输入

- `agent_id: int`
    
- `top_left: tuple[int, int]`
    
- `visible_agents: list[int]`
    

#### 作用

构造当前 agent 的 `blocking_map`

#### 输出

- `np.ndarray`
    
- shape = `(self.observation_size, self.observation_size)`
    
- dtype = `np.float32`
    

#### 第一版简化规则

采用 endpoint blocking：

- 若某条 corridor 中有其他 agent 当前占据
    
- 则该 corridor 的所有 endpoint 在 `blocking_map` 中置为 1
    
- 否则置为 0
    

#### 实现建议

对每个 endpoint：

1. 用 `endpoint_to_corridor` 找到 corridor id
    
2. 查看该 corridor 的 `corridor_cells`
    
3. 判断其中是否有除当前 agent 外的其他 agent 当前位置落在其中
    
4. 若有，则该 endpoint 在 FOV 内时置 1
    

注意：  
第一版先做简化版 blocking，不必实现 PRIMAL2 那种方向敏感规则。

---

## 六、最终需要改写 `_observe(self, agent_id)`

请在保留现有原始 4 通道构造逻辑的基础上，扩展 `_observe()`。

### 要求

1. 原始 4 通道逻辑尽量少改：
    
    - `poss_map`
        
    - `goal_map`
        
    - `goals_map`
        
    - `obs_map`
        
2. 在得到 `visible_agents` 后，新增：
    
    - `blocking_map = self._build_blocking_map(agent_id, top_left, visible_agents)`
        
    - `delta_x_map = self._extract_fov(self.delta_x_full, top_left, fill_value=0.0)`
        
    - `delta_y_map = self._extract_fov(self.delta_y_full, top_left, fill_value=0.0)`
        
    - `pred_1, pred_2, pred_3 = self._build_prediction_maps(agent_id, visible_agents, top_left, horizon=3)`
        
3. 目标向量 `[dx, dy, mag]` 保持现有逻辑不变
    
4. 最终返回 10 通道 + 目标向量
    

---

## 七、实现顺序要求

请按下面顺序完成，先不要改 `ACNet.py`：

1. 实现 `_is_free_cell`
    
2. 实现 `_get_free_neighbors`
    
3. 实现 `_is_corridor_cell`
    
4. 实现 `_find_corridor_components`
    
5. 实现 `_find_corridor_endpoints`
    
6. 实现 `_build_delta_maps`
    
7. 实现 `_compute_corridor_cache`
    
8. 实现 `_extract_fov`
    
9. 实现 `_predict_agent_future_positions`
    
10. 实现 `_build_prediction_maps`
    
11. 实现 `_build_blocking_map`
    
12. 改写 `_observe()`
    

---

## 八、代码风格要求

- 尽量保持与现有 `mapf_gym.py` 一致的命名和风格
    
- 优先做“清晰、稳定、能跑”的第一版
    
- 不要在第一版加入 pathlength map
    
- 不要在第一版实现过于复杂的 corridor convention
    
- blocking 先做简化版
    
- future prediction 直接使用现有 `getAstarCosts`
    

---

## 九、额外提醒

- 当前测试脚本 `primal_testing.py` 可能导入的是 `mapf_gym_cap.py`，不是 `mapf_gym.py`
    
- 所以这轮先专注改 `mapf_gym.py`
    
- 后续测试时再决定是否同步修改 `mapf_gym_cap.py` 或调整测试脚本导入
    
- 环境侧 10 通道确认无误后，再去改 `ACNet.py` 的输入通道数，从 4 改成 10
    

---

请先根据上述要求实现 `mapf_gym.py` 中的 helper methods 和 `_observe()` 改写。实现过程中优先保证逻辑正确和接口清晰，必要时可在关键位置添加简短注释。

````

你也可以在发给 Codex 前，再加一句：

```markdown
请先阅读当前文件中已有的 `MAPFEnv`、`_setWorld()`、`_observe()`、`getAstarCosts()` 实现，再按照上面的要求做最小改动。
````