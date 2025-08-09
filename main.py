"""
Minimal MAPPO (shared policy) for PettingZoo pursuit_v4 (parallel API).
Focus: MAPPO logic (masking, centralized critic, per-agent adv norm, separate LRs).
Run: python mappo_pursuit.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pettingzoo.sisl import pursuit_v4
import supersuit as ss
from collections import deque
import time

# ---------------------------
# Hyperparams (tune as needed)
# ---------------------------
NUM_STEPS = 128        # rollout length
NUM_UPDATES = 1000
NUM_EPOCHS = 4
MINI_BATCHES = 4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_PARAM = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 10

# ---------------------------
# Networks
# ---------------------------
class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        # obs: [batch, obs_dim]
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.logits(x)  # logits for Categorical


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, num_agents, hidden=HIDDEN):
        super().__init__()
        # We'll add an agent-id one-hot concatenated to global state
        self.input_dim = global_state_dim + num_agents
        self.fc1 = nn.Linear(self.input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, global_state, agent_id_onehot):
        # global_state: [batch, global_state_dim]
        # agent_id_onehot: [batch, num_agents]
        x = torch.cat([global_state, agent_id_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x).squeeze(-1)  # [batch]


# ---------------------------
# Helpers
# ---------------------------
def flatten_obs(obs):
    # obs: array shape (num_agents, H, W, C)
    # returns: flattened float32 numpy array [num_agents, obs_dim]
    num_agents = len(obs)
    flat = np.array([o.ravel() for o in obs], dtype=np.float32)
    return flat

def make_onehot(agent_idx, num_agents):
    v = np.zeros((num_agents,), dtype=np.float32)
    v[agent_idx] = 1.0
    return v

# ---------------------------
# Rollout buffer
# ---------------------------
class RolloutBuffer:
    def __init__(self, num_steps, num_agents, obs_dim):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        self.obs = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.masks = np.ones((num_steps, num_agents), dtype=np.float32)       # done mask (1 = not done)
        self.agent_masks = np.ones((num_steps, num_agents), dtype=np.float32) # alive mask (1 = active)
        self.global_states = np.zeros((num_steps, num_agents, obs_dim * num_agents), dtype=np.float32)

        self.ptr = 0

    def insert(self, obs_batch, actions_batch, logp_batch, rew_batch, done_mask_batch, agent_mask_batch, global_state_batch):
        t = self.ptr
        self.obs[t] = obs_batch
        self.actions[t] = actions_batch
        self.log_probs[t] = logp_batch
        self.rewards[t] = rew_batch
        self.masks[t] = done_mask_batch
        self.agent_masks[t] = agent_mask_batch
        self.global_states[t] = global_state_batch
        self.ptr += 1

    def reset(self):
        self.ptr = 0

# ---------------------------
# GAE
# ---------------------------
def compute_gae(rewards, masks, values, gamma=GAMMA, lam=LAMBDA):
    # rewards, masks, values: shape [T, N]
    T, N = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):

        ##  In general this is save cause we include the boostrap inside the values 
        next_values = values[t + 1] #  if t + 1 < T else np.zeros((N,), dtype=np.float32)
        delta = rewards[t] + gamma * next_values * masks[t] - values[t]
        lastgaelam = delta + gamma * lam * masks[t] * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:T]
    return advantages, returns

# ---------------------------
# Main training scaffolding
# ---------------------------
def train():
    # --- env
    env = pursuit_v4.parallel_env(max_cycles=300, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
    n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
    catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)


    env = ss.color_reduction_v0(env, mode='B')  # Simplify observations (optional)
    env = ss.resize_v1(env, x_size=16, y_size=16)  # Resize obs (optional)
    env = ss.frame_stack_v1(env, 3)  # Stack frames (optional)
    obs_dict = env.reset()
    agent_list = env.agents[:]  # e.g. ['pursuer_0'...]
    num_agents = len(agent_list)
    obs_example = obs_dict[0][agent_list[0]]
    obs_shape = obs_example.shape
    obs_dim = int(np.prod(obs_shape))
    action_space = env.action_space(agent_list[0])
    assert action_space.n is not None
    action_dim = action_space.n

    # --- networks + optimizers
    actor = SharedActor(obs_dim, action_dim).to(DEVICE)
    critic = CentralizedCritic(global_state_dim=obs_dim * num_agents, num_agents=num_agents).to(DEVICE)

    actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # --- buffers
    buffer = RolloutBuffer(NUM_STEPS, num_agents, obs_dim)

    ep_returns = deque(maxlen=100)
    total_steps = 0
    start_time = time.time()

    obs,_ = env.reset()  # dict keyed by agent
    # track alive status per agent (1 active, 0 inactive)
    agent_alive = {a: 1 for a in agent_list}
    accumulated_reward = {a: 0 for a in agent_list}
    for update in range(1, NUM_UPDATES + 1):
        # Collect rollouts
        buffer.reset()
        
        for step in range(NUM_STEPS):
            # build per-agent arrays
            obs_batch = np.zeros((num_agents, obs_dim), dtype=np.float32)
            actions_batch = np.zeros((num_agents,), dtype=np.int64)
            logp_batch = np.zeros((num_agents,), dtype=np.float32)
            agent_mask_batch = np.zeros((num_agents,), dtype=np.float32)
            reward_batch = np.zeros((num_agents,), dtype=np.float32)
            done_mask_batch = np.ones((num_agents,), dtype=np.float32)

            # flatten obs and collect actions
            obs_list = [obs[a] for a in agent_list]
            flat_obs = flatten_obs(obs_list)  # [num_agents, obs_dim]

            # global state is concatenation of flattened observations in agent order
            global_state = np.tile(flat_obs.ravel(), (num_agents, 1))  # repeated per agent
            # but we'll also feed agent onehots separately to critic

            for i, a in enumerate(agent_list):
                if obs[a] is None:
                    # agent is done/inactive
                    agent_mask_batch[i] = 0.0
                    actions_batch[i] = 0
                    logp_batch[i] = 0.0
                else:
                    agent_mask_batch[i] = 1.0
                    obs_tensor = torch.tensor(flat_obs[i:i+1], dtype=torch.float32, device=DEVICE)
                    logits = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().cpu().numpy().item()
                    logp = dist.log_prob(torch.tensor([action], device=DEVICE)).detach().cpu().numpy().item()
                    actions_batch[i] = int(action)
                    logp_batch[i] = float(logp)

            # step environment in parallel: build action dict
            action_dict = {agent_list[i]: int(actions_batch[i]) if agent_mask_batch[i] else None for i in range(num_agents)}
            next_obs, rewards, dones, truncations, infos  = env.step(action_dict)
           
            for a in agent_list:
                accumulated_reward[a] += rewards[a]
            for a in agent_list:
                dones[a] = dones.get(a, False) or truncations.get(a, False)

            # fill reward, done mask, agent masks
            for i, a in enumerate(agent_list):
                if not dones[a]:
                    reward_batch[i] = rewards.get(a, 0.0) or 0.0
                else:
                    reward_batch[i] = 0.0
                # PettingZoo uses True when agent is done; we want mask=0 if done else 1
                done_mask_batch[i] = 0.0 if dones.get(a, False) else 1.0
                # agent alive: if an agent is removed/terminated we set agent_mask to 0
                agent_mask_batch[i] = 0.0 if dones.get(a, False) else 1.0

            # store flattened global_state per agent (same content repeated)
            global_state_batch = np.zeros((num_agents, obs_dim * num_agents), dtype=np.float32)
            flat_concat = flat_obs.ravel()
            for i in range(num_agents):
                global_state_batch[i] = flat_concat  # simple concatenation global state

            buffer.insert(
                obs_batch=flat_obs,
                actions_batch=actions_batch,
                logp_batch=logp_batch,
                rew_batch=reward_batch,
                done_mask_batch=done_mask_batch,
                agent_mask_batch=agent_mask_batch,
                global_state_batch=global_state_batch
            )

            obs = next_obs
            # print(total_steps)
            # print(truncations)
            # print(dones)

            # if all(truncations.values()):
            #     print("holdoin")
            if all(dones.values()):  # Only reset if all agents are done
                # print("sdfsdfsdf")
                obs, _ = env.reset()
                agent_alive = {a: 1 for a in agent_list}
                episode_average_return = 0 
                for a in agent_list: 
                    episode_average_return += accumulated_reward[a]
                    accumulated_reward[a] = 0
                ep_returns.append(episode_average_return.item()/N)
                # accumulated_reward = {a: 0 for a in agent_list}

            if obs == {}:
                print("holding out for all agents to finish")
                

            total_steps += num_agents

            # optional: track episode returns when env reports 'terminal' via infos
            for a in agent_list:
                if done_mask_batch[0] == 0.0:
                    if infos.get(a, {}).get("episode"):
                        ep_returns.append(infos[a]["episode"]["r"])

        # --- After collection: compute values with critic
        T = buffer.num_steps
        N = buffer.num_agents

        # prepare values array shape [T, N+1] (we will need next_value for final)
        values = np.zeros((T, N), dtype=np.float32)
        with torch.no_grad():
            for t in range(T):
                # for each agent, compute critic value: pass global_state and agent-onehot
                gs = torch.tensor(buffer.global_states[t], dtype=torch.float32, device=DEVICE)  # [N, global_state_dim]
                # build agent id onehots
                id_oh = np.stack([make_onehot(i, N) for i in range(N)], axis=0).astype(np.float32)
                id_oh_t = torch.tensor(id_oh, dtype=torch.float32, device=DEVICE)
                v_t = critic(gs, id_oh_t)  # [N]
                values[t] = v_t.cpu().numpy()

            if buffer.global_states.shape[0] > 0:
                last_gs = torch.tensor(buffer.global_states[-1], dtype=torch.float32, device=DEVICE)
                last_id_oh = np.stack([make_onehot(i, N) for i in range(N)], axis=0).astype(np.float32)
                last_id_oh_t = torch.tensor(last_id_oh, dtype=torch.float32, device=DEVICE)
                bootstrap_value = critic(last_gs, last_id_oh_t).cpu().numpy()
            else:
                bootstrap_value = np.zeros((N,), dtype=np.float32)
                    
                


            

            # next values (bootstrap): use last global state repeated; if terminal, will be zeroed by masks
            # we approximate next_values as zeros here for simplicity; could compute from last obs if env non-terminal
            ## TODO: Fix it to the the next observation
            # next_values = np.zeros((N,), dtype=np.float32)
            


        # compute advantages & returns
        advs, returns = compute_gae(buffer.rewards, buffer.masks, np.vstack([values, bootstrap_value[np.newaxis,...]]))

        # per-agent advantage normalization
        # advs shape [T, N] -> normalize per agent (column-wise)
        # To make sure that the gradient inoact all agents equally 
        advs_norm = np.zeros_like(advs)
        for i in range(N):
            col = advs[:, i]
            m = col.mean()
            s = col.std()
            advs_norm[:, i] = (col - m) / (s + 1e-8)

        # flatten everything for minibatching
        obs_flat = buffer.obs.reshape(-1, obs_dim)                # [T*N, obs_dim]
        actions_flat = buffer.actions.flatten()                   # [T*N]
        old_logp_flat = buffer.log_probs.flatten()
        adv_flat = advs_norm.flatten()
        ret_flat = returns.flatten()
        agent_mask_flat = buffer.agent_masks.flatten()
        global_flat = buffer.global_states.reshape(-1, obs_dim * N)

        # convert to torch
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=DEVICE)
        actions_tensor = torch.tensor(actions_flat, dtype=torch.long, device=DEVICE)
        old_logp_tensor = torch.tensor(old_logp_flat, dtype=torch.float32, device=DEVICE)
        adv_tensor = torch.tensor(adv_flat, dtype=torch.float32, device=DEVICE)
        ret_tensor = torch.tensor(ret_flat, dtype=torch.float32, device=DEVICE)
        agent_mask_tensor = torch.tensor(agent_mask_flat, dtype=torch.float32, device=DEVICE)
        global_tensor = torch.tensor(global_flat, dtype=torch.float32, device=DEVICE)

        # --- PPO updates
        batch_size = obs_tensor.shape[0]
        batch_inds = np.arange(batch_size)
        minibatch_size = max(batch_size // MINI_BATCHES, 1)

        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = batch_inds[start:start + minibatch_size]
                if len(mb_inds) == 0:
                    continue

                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_old_logp = old_logp_tensor[mb_inds]
                mb_adv = adv_tensor[mb_inds]
                mb_ret = ret_tensor[mb_inds]
                mb_agent_mask = agent_mask_tensor[mb_inds]
                mb_global = global_tensor[mb_inds]

                # Actor forward
                logits = actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * mb_adv
                actor_loss = - (torch.min(surr1, surr2) * mb_agent_mask).sum() / (mb_agent_mask.sum() + 1e-8)

                # Critic forward: need agent ids for these minibatch rows
                # reconstruct agent indices from flattened index: idx % N
                agent_indices = (mb_inds % N)
                id_oh = np.stack([make_onehot(int(idx), N) for idx in agent_indices], axis=0)
                id_oh_t = torch.tensor(id_oh, dtype=torch.float32, device=DEVICE)
                value_pred = critic(mb_global, id_oh_t)
                critic_loss = ((mb_ret - value_pred) ** 2 * mb_agent_mask).sum() / (mb_agent_mask.sum() + 1e-8)

                # optimize
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optim.step()

        # logging
        if update % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg_return = np.mean(ep_returns) if len(ep_returns) > 0 else float("nan")

            print(f"Update {update}/{NUM_UPDATES}, steps {total_steps}, avg_return {avg_return:.3f}, elapsed {elapsed:.1f}s")

    print("Training finished.")

if __name__ == "__main__":
    train()
