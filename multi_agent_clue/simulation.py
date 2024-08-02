import numpy as np
from env import MultiAgentClue

from pymdp.agent import Agent
from pymdp import utils, maths

import matplotlib.pyplot as plt

import copy

def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """

    if not np.isclose(belief_dist.sum(), 1.0):
      raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()

def main():

    env = MultiAgentClue(grid_dims = [5, 3], starting_locs = [(0, 0), (4, 0)], clue_locations = [(1, 1), (3, 1)], reward_condition = 'BOTTOM')

    num_grid_points = np.prod(env.grid_dims)
    grid = np.arange(num_grid_points).reshape(env.grid_dims)
    it = np.nditer(grid, flags=["multi_index"])
    loc_list = []
    while not it.finished:
        loc_list.append(it.multi_index)
        it.iternext()

    num_states  = [num_grid_points, num_grid_points, len(env.REWARD_CONDITONS)]
    num_obs     = [ num_grid_points, num_grid_points, len(env.CLUE_NAMES), len(env.REWARD_NAMES)]


    # --- A : Observation Liklihood Matrix ---

    A_m_shapes  = [ [o_dim] + num_states for o_dim in num_obs]
    A_m         = utils.obj_array_zeros(A_m_shapes)

    # Agent Location Modality - depends only on the current location state
    for i in range(num_grid_points):

        A_m[0][:, :, i, 0] = np.eye(num_grid_points, num_grid_points)
        A_m[0][:, :, i, 1] = np.eye(num_grid_points, num_grid_points)

    # Other Agent Location Modality - depends only on their current location state
    for i in range(num_grid_points):

        A_m[1][:, i, :, 0] = np.eye(num_grid_points, num_grid_points)
        A_m[1][:, i, :, 1] = np.eye(num_grid_points, num_grid_points)

    # Clue Names Modality - depends on if both agents are in the clue locations

    A_m[2][0, :, :, :] = 1.0

    for i, reward_loc in enumerate(env.reward_locations):

        A_m[2][0, loc_list.index(env.clue_locations[0]), loc_list.index(env.clue_locations[1]), i] = 0.0
        A_m[2][i+1, loc_list.index(env.clue_locations[0]), loc_list.index(env.clue_locations[1]), i] = 1.0

        A_m[2][0, loc_list.index(env.clue_locations[1]), loc_list.index(env.clue_locations[0]), i] = 0.0
        A_m[2][i+1, loc_list.index(env.clue_locations[1]), loc_list.index(env.clue_locations[0]), i] = 1.0

    # Reward Names Modality - depends on if the agents are in the correct reward block
    A_m[3][0, :, :, :] = 1.0

    rew_top_idx = loc_list.index(env.reward_locations[0]) # linear index of the location of the "TOP" reward location
    rew_bott_idx = loc_list.index(env.reward_locations[1]) # linear index of the location of the "BOTTOM" reward location

    # fill out the contingencies when the agent is in the "TOP" reward location
    A_m[3][0, rew_top_idx, :, :] = 0.0
    A_m[3][1, rew_top_idx, :, 0] = 1.0
    A_m[3][2, rew_top_idx, :, 1] = 1.0

    # fill out the contingencies when the agent is in the "BOTTOM" `reward location
    A_m[3][0, rew_bott_idx, :, :] = 0.0
    A_m[3][1, rew_bott_idx, :, 1] = 1.0
    A_m[3][2, rew_bott_idx, :, 0] = 1.0

    # --- B : Transition Liklihood Matrix ---

    num_controls = [len(env.ACTIONS), 1, 1]

    B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

    B_m = utils.obj_array_zeros(B_f_shapes)

    for action_id, action_label in enumerate(env.ACTIONS):

        for curr_state, grid_location in enumerate(loc_list):

            y, x = grid_location

            if action_label == "STAY":
                next_x = x
                next_y = y

            elif action_label == "UP":
                next_y = y - 1 if y > 0 else y 
                next_x = x

            elif action_label == "DOWN":
                next_y = y + 1 if y < (env.grid_dims[0]-1) else y 
                next_x = x

            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x 
                next_y = y

            elif action_label == "RIGHT":
                next_x = x + 1 if x < (env.grid_dims[1]-1) else x 
                next_y = y

            new_location = (next_y, next_x)
            next_state = loc_list.index(new_location)
            B_m[0][next_state, curr_state, action_id] = 1.0

    # No idea of how the other agent location will change
    B_m[1][:, :, 0]   = np.full((num_states[1], num_states[1]), 1 / num_states[1])
    # B_m[1][:, :, 0]     = np.eye(num_states[1])

    # Reward location will not change
    B_m[2][:, :, 0] = np.eye(num_states[2])

    # --- C : Encoding the preferences over the observations ---

    C_m = utils.obj_array_zeros(num_obs)

    C_m[3][1] = 2.0 
    C_m[3][2] = -4.0

    # --- D : Priors over the state space ---

    agent_0_starting_loc = (0, 0)
    agent_1_starting_loc = (4, 0)

    D_m_0 = utils.obj_array_uniform(num_states)
    D_m_0[0] = utils.onehot(loc_list.index(agent_0_starting_loc), num_grid_points)

    D_m_1       = utils.obj_array_uniform(num_states)
    D_m_1[0]    = utils.onehot(loc_list.index(agent_1_starting_loc), num_grid_points)

    # --- Simulation ---

    agent_0 = Agent(A = A_m, B = B_m, C = C_m, D = D_m_0, policy_len = 3)
    agent_1 = Agent(A = A_m, B = B_m, C = C_m, D = D_m_1, policy_len = 3)

    loc_0, loc_1, clue_obs, reward_obs_0, reward_obs_1 = env.reset()

    history_of_locs_0 = [loc_0]
    history_of_locs_1 = [loc_1]


    agent_0_obs = [loc_list.index(loc_0), loc_list.index(loc_1), env.CLUE_NAMES.index(clue_obs), env.REWARD_NAMES.index(reward_obs_0)]
    agent_1_obs = [loc_list.index(loc_1), loc_list.index(loc_0), env.CLUE_NAMES.index(clue_obs), env.REWARD_NAMES.index(reward_obs_1)]
    
    T = 8 # number of total timesteps

    env.render()

    for t in range(T):

        print("-----------")

        agent_0.infer_states(agent_0_obs)
        agent_1.infer_states(agent_1_obs)
        
        q_pi_0, G_0 = agent_0.infer_policies()
        q_pi_1, G_1 = agent_1.infer_policies()
        print(f'Agent 0: {q_pi_0}')
        print(f'Agent 1: {q_pi_1}')

        chosen_action_0_id = agent_0.sample_action()
        chosen_action_1_id = agent_1.sample_action()

        movement_0_id = int(chosen_action_0_id[0])
        movement_1_id = int(chosen_action_1_id[0])

        choice_action_0 = env.ACTIONS[movement_0_id]
        choice_action_1 = env.ACTIONS[movement_1_id]

        print(f'Agent 0: Action at time {t}: {choice_action_0}')
        print(f'Agent 1: Action at time {t}: {choice_action_1}')

        loc_0, loc_1, clue_obs, reward_obs_0, reward_obs_1 = env.step([choice_action_0, choice_action_1])

        agent_0_obs = [loc_list.index(loc_0), loc_list.index(loc_1), env.CLUE_NAMES.index(clue_obs), env.REWARD_NAMES.index(reward_obs_0)]
        agent_1_obs = [loc_list.index(loc_1), loc_list.index(loc_0), env.CLUE_NAMES.index(clue_obs), env.REWARD_NAMES.index(reward_obs_1)]

        history_of_locs_0.append(loc_0)
        history_of_locs_1.append(loc_1)

        print(f'Agent 0: Grid location at time {t}: {loc_0}')
        print(f'Agent 1: Grid location at time {t}: {loc_1}')

        print(f'Clue Obs {t}: {clue_obs}')

        print(f'Agent 0: Reward at time {t}: {reward_obs_0}')
        print(f'Agent 1: Reward at time {t}: {reward_obs_1}')

        env.render()
        # plot_beliefs(qs_1[2], "Agent 2: Reward Location")


if __name__ == "__main__":
   main()