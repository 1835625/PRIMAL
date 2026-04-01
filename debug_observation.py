import numpy as np
# import matplotlib.pyplot as plt
import mapf_gym

CHANNEL_NAMES = [
    "poss", "goal", "goals", "obs",
    "blocking", "delta_x", "delta_y",
    "pred_1", "pred_2", "pred_3"
]

# def show_obs(obs, title_prefix="agent"):
#     for i, name in enumerate(CHANNEL_NAMES):
#         plt.figure()
#         plt.imshow(obs[i], cmap="gray")
#         plt.title(f"{title_prefix}_{name}")
#         plt.colorbar()

def print_env_debug(env, obs, goal_vec, agent_id=1):
    print("=" * 60)
    print(f"agent_id = {agent_id}")
    print("num channels =", len(obs))
    print("goal_vec =", goal_vec)
    print("corridor_id_map =")
    print(env.corridor_id_map)
    print("delta_x_full =")
    print(env.delta_x_full)
    print("delta_y_full =")
    print(env.delta_y_full)
    print("blocking_map =")
    print(obs[4])
    print("delta_x_map =")
    print(obs[5])
    print("delta_y_map =")
    print(obs[6])
    print("corridor_cells =", env.corridor_cells)
    print("corridor_endpoints =", env.corridor_endpoints)
    print("endpoint_to_corridor =", env.endpoint_to_corridor)
    print("predicted positions for agent 2 =", env._predict_agent_future_positions(2, horizon=3))
    print("pred_1 =")
    print(obs[7])
    print("pred_2 =")
    print(obs[8])
    print("pred_3 =")
    print(obs[9])

def run_case(case_name, world, goals, num_agents, observe_agent=1):
    env = mapf_gym.MAPFEnv(
        num_agents=num_agents,
        observation_size=10,
        world0=world,
        goals0=goals,
        DIAGONAL_MOVEMENT=False
    )
    obs, goal_vec = env._observe(observe_agent)
    print(f"\n===== {case_name} =====")
    print("world =")
    print(world)
    print("goals =")
    print(goals)
    print_env_debug(env, obs, goal_vec, observe_agent)
    # show_obs(obs, f"{case_name}_agent{observe_agent}")

def make_case1_straight_corridor():
    world = -np.ones((10, 10), dtype=int)
    goals = np.zeros((10, 10), dtype=int)

    # 左侧房间
    world[3:6, 1:3] = 0
    # 右侧房间
    world[3:6, 7:9] = 0
    # 中间水平 corridor
    world[4, 3:7] = 0

    # agent 1 在左房间
    world[4, 1] = 1
    # goal 1 在右房间
    goals[4, 8] = 1

    return world, goals

def make_case2_blocking_corridor():
    world = -np.ones((10, 10), dtype=int)
    goals = np.zeros((10, 10), dtype=int)

    # 左侧房间
    world[3:6, 1:3] = 0
    # 右侧房间
    world[3:6, 7:9] = 0
    # 中间水平 corridor
    world[4, 3:7] = 0

    # agent 1：观察对象，在左房间
    world[4, 1] = 1
    goals[4, 8] = 1

    # agent 2：堵在 corridor 内
    world[4, 5] = 2
    goals[4, 7] = 2

    return world, goals

def make_case3_turning_prediction():
    world = -np.ones((10, 10), dtype=int)
    goals = np.zeros((10, 10), dtype=int)

    # carve 一个较大的自由区域
    world[1:9, 1:9] = 0

    # 中间放一堵竖墙，只留一个缺口
    world[2:8, 5] = -1
    world[6, 5] = 0   # 缺口，允许绕行

    # agent 1：观察对象，放左上
    world[2, 2] = 1
    goals[2, 3] = 1

    # agent 2：被预测对象，放在墙左侧
    world[4, 3] = 2
    # goal 2：放在墙右侧，迫使其绕到缺口再过去
    goals[4, 7] = 2

    return world, goals

if __name__ == "__main__":
    # # case 1
    # world, goals = make_case1_straight_corridor()
    # run_case("case1_straight_corridor", world, goals, num_agents=1, observe_agent=1)

    # # case 2
    # world, goals = make_case2_blocking_corridor()
    # run_case("case2_blocking_corridor", world, goals, num_agents=2, observe_agent=1)

    # case 3
    world, goals = make_case3_turning_prediction()
    run_case("case3_turning_prediction", world, goals, num_agents=2, observe_agent=1)

    # plt.show()