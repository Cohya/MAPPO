from pettingzoo.sisl import pursuit_v4
  # Predator-Prey environment
import supersuit as ss

# Create env
env = pursuit_v4.parallel_env(max_cycles=100, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)


env = ss.color_reduction_v0(env, mode='B')  # Simplify observations (optional)
env = ss.resize_v1(env, x_size=32, y_size=32)  # Resize obs (optional)
env = ss.frame_stack_v1(env, 3)  # Stack frames (optional)
observations = env.reset()

print("Agents:", env.agents)

for step in range(100):  # Run for 100 steps
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # Random actions for all agents
    # observations, rewards, dones, infos = env.step(actions)
    observations, rewards, dones, truncations, infos = env.step(actions)
    # Optional: print some information about the step
    print(f"Step {step}")
    print("Rewards:", rewards)
    
    if all(dones.values()):  # If all agents are done
        break
