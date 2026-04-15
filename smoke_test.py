from ACNet import ACNet
import tensorflow as tf
import numpy as np
import mapf_gym as mapf_gym

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

GRID_SIZE = 10
OBS_CHANNELS = 10
A_SIZE = 5
GLOBAL_NET_SCOPE = "global"

env = mapf_gym.MAPFEnv(
    num_agents=4,
    observation_size=GRID_SIZE,
    DIAGONAL_MOVEMENT=False,
    SIZE=(10, 10),
    PROB=(0.0, 0.001)
)

batch_obs = []
batch_goal = []
batch_valid = []

for agent_id in range(1, env.num_agents + 1):
    obs, goal = env._observe(agent_id)
    obs = np.asarray(obs, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)

    valid = np.zeros((A_SIZE,), dtype=np.float32)
    valid[env._listNextValidActions(agent_id)] = 1.0

    batch_obs.append(obs)
    batch_goal.append(goal)
    batch_valid.append(valid)

batch_obs = np.stack(batch_obs, axis=0)
batch_goal = np.stack(batch_goal, axis=0)
batch_valid = np.stack(batch_valid, axis=0)

print("batch_obs:", batch_obs.shape)
print("batch_goal:", batch_goal.shape)
print("batch_valid:", batch_valid.shape)

trainer = tf.train.AdamOptimizer(1e-4)

master_network = ACNet(
    GLOBAL_NET_SCOPE, A_SIZE, None, False,
    GRID_SIZE, GLOBAL_NET_SCOPE, obs_channels=OBS_CHANNELS
)

local_network = ACNet(
    "worker_1", A_SIZE, trainer, True,
    GRID_SIZE, GLOBAL_NET_SCOPE, obs_channels=OBS_CHANNELS
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    rnn_state0 = local_network.state_init

    feed_dict = {
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
        feed_dict=feed_dict
    )

    print("policy shape:", policy.shape)
    print("value shape:", value.shape)
    print("blocking shape:", blocking.shape)
    print("valids shape:", valids.shape)
    print("state_out[0] shape:", state_out[0].shape)
    print("state_out[1] shape:", state_out[1].shape)

    assert policy.shape == (batch_obs.shape[0], A_SIZE)
    assert value.shape == (batch_obs.shape[0], 1)
    assert blocking.shape == (batch_obs.shape[0], 1)
    assert valids.shape == (batch_obs.shape[0], A_SIZE)
    assert state_out[0].shape[0] == 1
    assert state_out[1].shape[0] == 1

    assert np.all(np.isfinite(policy))
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(blocking))
    assert np.all(np.isfinite(valids))

    print("Step 2 passed.")