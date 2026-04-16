import numpy as np
import mapf_gym

CHANNEL_NAMES = [
    "poss", "goal", "goals", "obs",
    "blocking", "delta_x", "delta_y",
    "pred_1", "pred_2", "pred_3"
]

def print_structure_debug(env, obs, goal_vec, case_name, agent_id=1):
    print("\n" + "=" * 80)
    print(f"CASE: {case_name}")
    print(f"observe agent = {agent_id}")
    print("goal_vec =", goal_vec)

    print("\nworld.state =")
    print(env.world.state)

    print("\ncorridor_id_map =")
    print(env.corridor_id_map)

    if hasattr(env, "corridor_type_map"):
        print("\ncorridor_type_map =")
        print(env.corridor_type_map)

    print("\ndelta_x_full =")
    print(env.delta_x_full)

    print("\ndelta_y_full =")
    print(env.delta_y_full)

    print("\nblocking_map =")
    print(obs[4])

    print("\ndelta_x_map =")
    print(obs[5])

    print("\ndelta_y_map =")
    print(obs[6])

    print("\ncorridor_cells =")
    print(env.corridor_cells)

    print("\ncorridor_endpoints =")
    print(env.corridor_endpoints)

    if hasattr(env, "corridor_stopping_points"):
        print("\ncorridor_stopping_points =")
        print(env.corridor_stopping_points)

    if hasattr(env, "corridor_ordered_cells"):
        print("\ncorridor_ordered_cells =")
        print(env.corridor_ordered_cells)

    if hasattr(env, "stopping_point_meta"):
        print("\nstopping_point_meta =")
        for k, v in env.stopping_point_meta.items():
            print(k, "->", v)


def run_case(case_name, world, goals, num_agents, observe_agent=1, obs_size=11):
    env = mapf_gym.MAPFEnv(
        num_agents=num_agents,
        observation_size=obs_size,
        world0=world.copy(),
        goals0=goals.copy(),
        DIAGONAL_MOVEMENT=False
    )
    obs, goal_vec = env._observe(observe_agent)
    print_structure_debug(env, obs, goal_vec, case_name, observe_agent)
    return env, obs, goal_vec


def make_case1_straight_corridor_clear():
    """
    左右两个房间，中间一条水平直走廊，无阻塞。
    预期重点：
    - corridor 应被识别为一条主要 corridor
    - 两端应有 stopping point
    - delta_x_map / delta_y_map 在 stopping point 上有稀疏值
    - blocking_map 应基本为 0
    """
    world = -np.ones((11, 11), dtype=int)
    goals = np.zeros((11, 11), dtype=int)

    world[3:8, 1:3] = 0      # 左房间
    world[3:8, 8:10] = 0     # 右房间
    world[5, 3:8] = 0        # 中间走廊

    world[5, 1] = 1          # 观察 agent
    goals[5, 9] = 1

    return world, goals, 1, 1


def make_case2_straight_corridor_near_block():
    """
    直走廊内已有其他 agent，且靠近观察侧。
    预期重点：
    - 当前侧 stopping point 的 blocking_map 应出现强阻塞（通常是 1）
    """
    world = -np.ones((11, 11), dtype=int)
    goals = np.zeros((11, 11), dtype=int)

    world[3:8, 1:3] = 0
    world[3:8, 8:10] = 0
    world[5, 3:8] = 0

    world[5, 1] = 1
    goals[5, 9] = 1

    world[5, 4] = 2          # 靠近左侧入口
    goals[5, 8] = 2

    return world, goals, 2, 1


def make_case3_straight_corridor_far_endpoint_occupied():
    """
    远端 endpoint 一侧被占用，用来观察 soft caution。
    预期重点：
    - 可能在当前侧 stopping point 上看到 -1
    - 用来验证 blocking_map 的三值语义
    """
    world = -np.ones((11, 11), dtype=int)
    goals = np.zeros((11, 11), dtype=int)

    world[3:8, 1:3] = 0
    world[3:8, 8:10] = 0
    world[5, 3:8] = 0

    world[5, 1] = 1
    goals[5, 9] = 1

    world[5, 8] = 2          # 远端房间/端点附近
    goals[5, 1] = 2

    return world, goals, 2, 1


def make_case4_dead_end_corridor():
    """
    单入口死胡同。
    预期重点：
    - dead-end corridor 应被识别
    - delta_x / delta_y 应为 0 或非常稀疏
    - 若死胡同里有别的 agent，入口 stopping point 应显示强阻塞
    """
    world = -np.ones((11, 11), dtype=int)
    goals = np.zeros((11, 11), dtype=int)

    world[4:8, 1:4] = 0      # 左侧房间
    world[5, 4:8] = 0        # 向右延申的死胡同

    world[5, 2] = 1
    goals[5, 3] = 1

    world[5, 6] = 2          # 死胡同内部 agent
    goals[5, 7] = 2

    return world, goals, 2, 1


def make_case5_t_junction():
    """
    T 字路口。
    预期重点：
    - corridor 是否被合理切分
    - stopping point 是否落在合理位置
    - 不要求 blocking 特别明显，重点看 corridor 结构
    """
    world = -np.ones((11, 11), dtype=int)
    goals = np.zeros((11, 11), dtype=int)

    # 主水平走廊
    world[5, 2:9] = 0
    # 竖直支路
    world[2:6, 5] = 0

    # 三个端部稍微扩一下，形成决策区
    world[4:7, 1:3] = 0
    world[4:7, 8:10] = 0
    world[1:3, 4:7] = 0

    world[5, 2] = 1
    goals[2, 5] = 1

    return world, goals, 1, 1


if __name__ == "__main__":
    cases = [
        ("case1_straight_corridor_clear", make_case1_straight_corridor_clear),
        ("case2_straight_corridor_near_block", make_case2_straight_corridor_near_block),
        ("case3_straight_corridor_far_endpoint_occupied", make_case3_straight_corridor_far_endpoint_occupied),
        ("case4_dead_end_corridor", make_case4_dead_end_corridor),
        ("case5_t_junction", make_case5_t_junction),
    ]

    for case_name, fn in cases:
        world, goals, num_agents, observe_agent = fn()
        run_case(case_name, world, goals, num_agents, observe_agent, obs_size=11)