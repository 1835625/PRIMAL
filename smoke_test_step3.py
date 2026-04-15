# smoke_test_step3.py
# 用于检查：
# 1) 10 通道 observation 能否进入 ACNet
# 2) goal_pos / train_valid / target_blockings 等训练接口是否对齐
# 3) 单次 loss 计算和 apply_grads 是否可以正常执行
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import numpy as np
import tensorflow as tf

import mapf_gym as mapf_gym
from ACNet import ACNet


def build_env(grid_size=10, num_agents=4):
    """
    构造一个简单环境：
    - 静态地图为空白
    - agent / goal 由 blank_world=True 随机放置
    这样可以避免 PROB=(0,0) 时 triangular 的报错
    """
    world0 = np.zeros((grid_size, grid_size), dtype=int)

    env = mapf_gym.MAPFEnv(
        num_agents=num_agents,
        observation_size=grid_size,
        world0=world0,
        goals0=None,
        DIAGONAL_MOVEMENT=False,
        blank_world=True
    )
    return env


def collect_batch(env, a_size):
    """
    为每个 agent 收集一条 observation / goal / valid-action label
    """
    batch_obs = []
    batch_goal = []
    batch_valid = []

    for agent_id in range(1, env.num_agents + 1):
        obs, goal = env._observe(agent_id)

        obs = np.asarray(obs, dtype=np.float32)     # (C, H, W)
        goal = np.asarray(goal, dtype=np.float32)   # (3,)

        valid = np.zeros((a_size,), dtype=np.float32)
        valid_actions = env._listNextValidActions(agent_id)
        valid[valid_actions] = 1.0

        batch_obs.append(obs)
        batch_goal.append(goal)
        batch_valid.append(valid)

    batch_obs = np.stack(batch_obs, axis=0)     # (B, C, H, W)
    batch_goal = np.stack(batch_goal, axis=0)   # (B, 3)
    batch_valid = np.stack(batch_valid, axis=0) # (B, A)

    return batch_obs, batch_goal, batch_valid


def main():
    # -----------------------------
    # 超参数
    # -----------------------------
    GRID_SIZE = 10
    OBS_CHANNELS = 10
    NUM_AGENTS = 4
    A_SIZE = 5                  # 非对角移动时动作数：still, N, E, S, W
    GLOBAL_NET_SCOPE = "global"

    # -----------------------------
    # 固定随机种子，方便复现
    # -----------------------------
    np.random.seed(0)
    random.seed(0)
    tf.reset_default_graph()
    tf.set_random_seed(0)

    # -----------------------------
    # 构造环境并收集一个 mini-batch
    # -----------------------------
    env = build_env(grid_size=GRID_SIZE, num_agents=NUM_AGENTS)
    batch_obs, batch_goal, batch_valid = collect_batch(env, A_SIZE)

    print("batch_obs shape   :", batch_obs.shape)
    print("batch_goal shape  :", batch_goal.shape)
    print("batch_valid shape :", batch_valid.shape)

    assert batch_obs.shape == (NUM_AGENTS, OBS_CHANNELS, GRID_SIZE, GRID_SIZE)
    assert batch_goal.shape == (NUM_AGENTS, 3)
    assert batch_valid.shape == (NUM_AGENTS, A_SIZE)

    # -----------------------------
    # 建图：global net + local net
    # -----------------------------
    trainer = tf.train.AdamOptimizer(1e-4)

    # global net: 只为了提供 global_vars
    master_network = ACNet(
        scope=GLOBAL_NET_SCOPE,
        a_size=A_SIZE,
        trainer=None,
        TRAINING=False,
        GRID_SIZE=GRID_SIZE,
        GLOBAL_NET_SCOPE=GLOBAL_NET_SCOPE,
        obs_channels=OBS_CHANNELS
    )

    # local net: 真正执行前向和反向
    local_network = ACNet(
        scope="worker_1",
        a_size=A_SIZE,
        trainer=trainer,
        TRAINING=True,
        GRID_SIZE=GRID_SIZE,
        GLOBAL_NET_SCOPE=GLOBAL_NET_SCOPE,
        obs_channels=OBS_CHANNELS
    )

    # -----------------------------
    # 打开 session，执行一次前向 + 一次反向
    # -----------------------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        B = batch_obs.shape[0]
        rnn_state0 = local_network.state_init

        # 构造一组假的训练标签
        # 这里只是 smoke test，不追求语义正确，只要 shape / dtype / 图连接正确即可
        actions = np.array(
            [np.where(batch_valid[i] > 0)[0][0] for i in range(B)],
            dtype=np.int32
        )
        target_v = np.zeros((B,), dtype=np.float32)
        advantages = np.ones((B,), dtype=np.float32)
        train_value = np.ones((B,), dtype=np.float32)
        target_blockings = np.zeros((B,), dtype=np.float32)

        # ---------- 先做一次前向 ----------
        forward_feed_dict = {
            local_network.inputs: batch_obs,
            local_network.goal_pos: batch_goal,
            local_network.state_in[0]: rnn_state0[0],
            local_network.state_in[1]: rnn_state0[1],
        }

        policy, value, blocking, valids, state_out = sess.run(
            [
                local_network.policy,
                local_network.value,
                local_network.blocking,
                local_network.valids,
                local_network.state_out,
            ],
            feed_dict=forward_feed_dict
        )

        print("policy shape      :", policy.shape)
        print("value shape       :", value.shape)
        print("blocking shape    :", blocking.shape)
        print("valids shape      :", valids.shape)
        print("state_out[0] shape:", state_out[0].shape)
        print("state_out[1] shape:", state_out[1].shape)

        assert policy.shape == (B, A_SIZE)
        assert value.shape == (B, 1)
        assert blocking.shape == (B, 1)
        assert valids.shape == (B, A_SIZE)
        assert state_out[0].shape == (1, 512)
        assert state_out[1].shape == (1, 512)

        assert np.all(np.isfinite(policy))
        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(blocking))
        assert np.all(np.isfinite(valids))

        # ---------- 再做一次反向 ----------
        train_feed_dict = {
            local_network.inputs: batch_obs,
            local_network.goal_pos: batch_goal,
            local_network.actions: actions,
            local_network.train_valid: batch_valid,
            local_network.target_v: target_v,
            local_network.advantages: advantages,
            local_network.train_value: train_value,
            local_network.target_blockings: target_blockings,
            local_network.state_in[0]: rnn_state0[0],
            local_network.state_in[1]: rnn_state0[1],
        }

        outputs = sess.run(
            [
                local_network.value_loss,
                local_network.policy_loss,
                local_network.valid_loss,
                local_network.entropy,
                local_network.blocking_loss,
                local_network.grad_norms,
                local_network.var_norms,
                local_network.apply_grads,
            ],
            feed_dict=train_feed_dict
        )

        metric_names = [
            "value_loss",
            "policy_loss",
            "valid_loss",
            "entropy",
            "blocking_loss",
            "grad_norms",
            "var_norms",
        ]

        print("\nStep 3 metrics:")
        for name, val in zip(metric_names, outputs[:-1]):
            print("{:>14s}: {}".format(name, float(val)))
            assert np.isfinite(val), "{} is not finite!".format(name)

        print("\nStep 3 passed.")


if __name__ == "__main__":
    main()