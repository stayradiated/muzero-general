import datetime
import os

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 4, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(64))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 16  # Maximum number of moves if game is not finished before
        self.num_simulations = 32  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 32  # Number of game moves to keep for every batch element
        self.td_steps = 32  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = PetesGame()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                choice = int(
                    input(
                        f"Enter the action (0-31) to play for the player {self.to_play()}: "
                    )
                )
                if (
                    choice in self.legal_actions()
                    and 0 <= choice
                    and choice < 64
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(self.env.parse_action(action_number))


class PetesGame:
    def __init__(self):
        self.size = 4
        self.reset()

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.size, self.size), dtype="int32")
        self.player = 1
        return numpy.array([self.board], dtype="int32")

    def parse_action(self, action):
        if action < 16:
            row = action // self.size
            col = action % self.size
            return (row,1, col,1)
        elif action < 28:
            action -= 16
            row = action // (self.size - 1)
            col = action % (self.size - 1)
            return(row,1, col,2)
        elif action < 36:
            action -= 28
            row = action // (self.size - 2)
            col = action % (self.size - 2)
            return (row,1, col,3)
        elif action < 40:
            action -= 36
            row = action // (self.size - 3)
            col = action % (self.size - 3)
            return (row,1, col,4)
        elif action < 52:
            action -= 40
            row = action // self.size
            col = action % self.size
            return (row,2, col,1)
        elif action < 60:
            action -= 52
            row = action // self.size
            col = action % self.size
            return (row,3, col,1)
        elif action < 64:
            action -= 60
            row = action // self.size
            col = action % self.size
            return (row,4, col,1)

    def step(self, action):
        (r1,r2,c1,c2) = self.parse_action(action)
        self.board[r1:r1+r2,c1:c1+c2] = 1

        free_cells = numpy.count_nonzero(numpy.where(self.board == 0, 1, 0))

        done = 0 if free_cells > 0 else 1

        if free_cells == 0:
            reward = -1
        elif free_cells == 1:
            reward = 1
        else:
            reward = 0

        self.player *= -1

        return numpy.array([self.board], dtype="int32"), reward, done

    def legal_actions(self):
        actions = []

        offset = 0

        # 1x1 cells
        for i in range(self.size * self.size):
            row = i // self.size
            col = i % self.size
            if self.board[row, col] == 0:
                actions.append(i + offset)
        offset += (self.size * self.size)

        # 1x2 rows
        for i in range((self.size - 1) * self.size):
            row = i // (self.size - 1)
            col = i % (self.size - 1)
            if numpy.all(self.board[row, col:col+2] == 0):
                actions.append(i + offset)
        offset += ((self.size - 1)* self.size)

        # 1x3 rows
        for i in range((self.size - 2) * self.size):
            row = i // (self.size - 2)
            col = i % (self.size - 2)
            if numpy.all(self.board[row, col:col+3] == 0):
                actions.append(i + offset)
        offset += ((self.size - 2)* self.size)

        # 1x4 rows
        for i in range((self.size - 3) * self.size):
            row = i // (self.size - 3)
            col = i % (self.size - 3)
            if numpy.all(self.board[row, col:col+4] == 0):
                actions.append(i + offset)
        offset += ((self.size - 3)* self.size)

        # 2x1 cols
        for i in range((self.size - 1) * self.size):
            row = i // self.size
            col = i % self.size
            if numpy.all(self.board[row:row+2, col] == 0):
                actions.append(i + offset)
        offset += ((self.size - 1)* self.size)

        # 3x1 cols
        for i in range((self.size - 2) * self.size):
            row = i // self.size
            col = i % self.size
            if numpy.all(self.board[row:row+3, col] == 0):
                actions.append(i + offset)
        offset += ((self.size - 2) * self.size)

        # 4x1 cols
        for i in range((self.size - 3) * self.size):
            row = i // self.size
            col = i % self.size
            if numpy.all(self.board[row:row+4, col] == 0):
                actions.append(i + offset)
        offset += ((self.size - 3) * self.size)

        # columns
        return actions

    def render(self):
        board = ""
        for x in range(self.size):
            for y in range(self.size):
                cell = self.board[x,y]
                if cell == 0:
                    board += '.'
                else:
                    board += '░'
            board += "\n"
        print(board)
