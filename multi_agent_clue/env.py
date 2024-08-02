from pymdp.envs import Env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

class MultiAgentClue(Env):

    ACTIONS             = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    CLUE_NAMES          = ['NULL', 'REWARD_ON_TOP', 'REWARD_ON_BOTTOM']
    REWARD_NAMES        = ['NULL', 'CHEESE', 'SHOCK']
    
    REWARD_CONDITONS    = ["TOP", "BOTTOM"]

    def __init__(self, grid_dims = [5, 3], starting_locs = [(0, 0), (4, 0)], clue_locations = [(1, 1), (3, 1)], reward_condition = 'TOP'):

        self.grid_dims          = grid_dims

        self.reward_locations        = [(0, grid_dims[1] - 1), (grid_dims[0] - 1, grid_dims[1] - 1)]

        self.init_loc_0         = starting_locs[0]
        self.init_loc_1         = starting_locs[1]

        self.current_locations  = [self.init_loc_0, self.init_loc_1]

        self.clue_locations           = clue_locations

        self.reward_condition = reward_condition


    def reset(self):

        self.current_locations  = [self.init_loc_0, self.init_loc_1]

        loc_obs_0 = self.init_loc_0
        loc_obs_1 = self.init_loc_1

        if self.current_locations[0] == self.clue_locations[0] and self.current_locations[1] == self.clue_locations[1] or self.current_locations[0] == self.clue_locations[1] and self.current_locations[1] == self.clue_locations[0]:
          clue_obs = self.CLUE_NAMES[self.REWARD_CONDITONS.index(self.REWARD_CONDITONS)+1]
        else:
          clue_obs = 'NULL'
          
        reward_obs = 'NULL'

        return loc_obs_0, loc_obs_1, clue_obs, reward_obs, reward_obs
    
    def step(self,action_labels):
        
        for i, action_label in enumerate(action_labels):

          (Y, X) = self.current_locations[i]

          if action_label == "STAY":
            Y_new, X_new = Y, X

          elif action_label == "UP": 
            
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

          elif action_label == "DOWN": 

            Y_new = Y + 1 if Y < (self.grid_dims[0]-1) else Y
            X_new = X

          elif action_label == "LEFT": 
            Y_new = Y
            X_new = X - 1 if X > 0 else X

          elif action_label == "RIGHT":
            Y_new = Y
            X_new = X +1 if X < (self.grid_dims[1]-1) else X
          
          self.current_locations[i] = (Y_new, X_new)

        loc_obs_0 = self.current_locations[0] 
        loc_obs_1 = self.current_locations[1]

        # Clue Obs

        if self.current_locations[0] == self.clue_locations[0] and self.current_locations[1] == self.clue_locations[1] or self.current_locations[0] == self.clue_locations[1] and self.current_locations[1] == self.clue_locations[0]:
          
          clue_obs = self.CLUE_NAMES[self.REWARD_CONDITONS.index(self.reward_condition) + 1]

        else:
          clue_obs = self.CLUE_NAMES[0]

          # Reward Obs

        if self.current_locations[0] == self.reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs_0 = 'CHEESE'
          else:
            reward_obs_0 = 'SHOCK'
        elif self.current_locations[0] == self.reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs_0 = 'CHEESE'
          else:
            reward_obs_0 = 'SHOCK'
        else:
          reward_obs_0 = 'NULL'

        if self.current_locations[1] == self.reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs_1 = 'CHEESE'
          else:
            reward_obs_1 = 'SHOCK'
        elif self.current_locations[1] == self.reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs_1 = 'CHEESE'
          else:
            reward_obs_1 = 'SHOCK'
        else:
          reward_obs_1 = 'NULL'

        return loc_obs_0, loc_obs_1, clue_obs, reward_obs_0, reward_obs_1
    
    def render(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the grid visualization
        X, Y = np.meshgrid(np.arange(self.grid_dims[1] + 1), np.arange(self.grid_dims[0] + 1))
        h = ax.pcolormesh(X, Y, np.ones(self.grid_dims), edgecolors='k', vmin=0, vmax=30, linewidth=3, cmap='coolwarm')
        ax.invert_yaxis()

        # Put gray boxes around the possible reward locations
        for loc in self.reward_locations:
            ax.add_patch(patches.Rectangle((loc[1], loc[0]), 1.0, 1.0, linewidth=5, edgecolor='gray', facecolor='none'))

        text_offsets = [0.4, 0.6]

        clue_grid = np.ones(self.grid_dims)
        for loc in self.clue_locations:
            clue_grid[loc[0], loc[1]] = 15.0

        # Highlight the clue and reward locations
        # for loc in self.reward_locations:
        #     row_coord, column_coord = loc
        #     clue_grid[row_coord, column_coord] = 5.0
        #     ax.text(column_coord + text_offsets[0], row_coord + text_offsets[1], "R", fontsize=15, color='k', fontweight='bold')

        reward_top = ax.add_patch(patches.Rectangle((self.reward_locations[0][1], self.reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
        reward_bottom = ax.add_patch(patches.Rectangle((self.reward_locations[1][1], self.reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))

        reward_loc = self.reward_locations[0] if self.reward_condition == "TOP" else self.reward_locations[1]

        if self.reward_condition == "TOP":
            reward_top.set_edgecolor('g')
            reward_top.set_facecolor('g')
            reward_bottom.set_edgecolor([0.7, 0.2, 0.2])
            reward_bottom.set_facecolor([0.7, 0.2, 0.2])

        elif self.reward_condition == "BOTTOM":
            reward_bottom.set_edgecolor('g')
            reward_bottom.set_facecolor('g')
            reward_top.set_edgecolor([0.7, 0.2, 0.2])
            reward_top.set_facecolor([0.7, 0.2, 0.2])
        reward_top.set_zorder(1)
        reward_bottom.set_zorder(1)

        # Update the color map array
        h.set_array(clue_grid.ravel())

        # Plot the clue locations
        # for loc in self.clue_locations:
        #     ax.scatter(loc[1] + 0.5, loc[0] + 0.5, color='blue', s=200, label="Clue Location", zorder=5)

        # Plot the agent locations
        for loc in self.current_locations:
            ax.scatter(loc[1] + 0.5, loc[0] + 0.5, color='red', s=200, marker='x', label="Agent Location", zorder=6)

        # Add a legend
        # plt.legend(loc='upper right')
        plt.title("Grid with Clue, Reward, and Agent Locations", fontsize=18, fontweight='bold')
        plt.show()