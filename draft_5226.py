import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import random

statespace_size_g = 49
l2_g = 1024
l3_g = 512
l4_g = 256
l5_g = 128
l6_g = 64

def same_location(onehot1, onehot2):
    """Compares two one-hot encoded locations."""
    return np.array_equal(onehot1, onehot2)

def location_to_onehot(loc, height, width):
    """Converts a 2D location [x, y] to a one-hot encoded vector."""
    index = loc[0] * width + loc[1]
    onehot = torch.zeros(height * width)
    onehot[index] = 1
    return onehot

def onehot_to_location(onehot, width):
    """Converts a one-hot encoded vector back to a 2D location [x, y]."""
    index = np.argmax(onehot)
    return [index // width, index % width]


class Environment:
    def __init__(self, height, width):
        """
        :param height: The height of the environment grid.
        :param width: The width of the environment grid.
        :functionality: Initializes the Environment object with height and width,
                        and calls regenerate_locations to set random target, agent, and item locations.
        """
        self.item_location = None
        self.agent_location = None
        self.width = width
        self.height = height
        self.target_location = None
        self.regenerate_locations()

    def regenerate_locations(self):
        """
        :return: None
        :functionality: Generates new random locations for the target, agent, and the item,
                        ensuring no overlaps between them.
        """
        self.set_target_location(self.generate_random_location())
        self.set_agent_location(self.generate_random_location([self.target_location]))
        self.set_item_location(self.generate_random_location([self.target_location, self.agent_location]))

    def set_target_location(self, new_location):
        """
        :param new_location: The new location for the target.
        :return: None
        :functionality: Updates the target's location to the new location.
        """
        self.target_location = new_location

    def set_agent_location(self, new_location):
        """
        :param new_location: The new location for the agent.
        :return: None
        :functionality: Updates the agent's location to the new location.
        """
        self.agent_location = new_location

    def set_item_location(self, new_location):
        """
        :param new_location: The new location for the item.
        :return: None
        :functionality: Updates the item's location to the new location.
        """
        self.item_location = new_location

    def generate_random_location(self, exclude=()):
        """
        :param exclude: A list of locations to be excluded when generating a new random location.
        :return: A new random location as a list [x, y] that is not in the exclude list.
        :functionality: Generates a random location within the bounds of the environment
                        grid, ensuring that it does not match any of the locations in the exclude list.
        """

        # Helper function to check if a location already exists in the exclude list
        def location_exist(new_location):
            for loc in exclude:
                if same_location(loc, new_location):
                    return True
            return False

        # Generate an initial random location
        location = [np.random.randint(0, self.height), np.random.randint(0, self.width)]
        onehot_location = location_to_onehot(location, self.height, self.width)

        # If the generated location exists in the exclude list, keep generating new ones until it doesn't
        while location_exist(onehot_location):
            location = [np.random.randint(0, self.height), np.random.randint(0, self.width)]
            onehot_location = location_to_onehot(location, self.height, self.width)

        return onehot_location

    def get_reward(self, state):
        """
        :param state: The current state of the agent.
        :return: None
        :functionality: Evaluates the agent's current state to assign a reward.
                        Updates the state's reward attribute accordingly.
        """
        # If the agent is at the item location and not carrying the item, reward it and update its state
        if same_location(state.agent_location, self.item_location) and not state.carry_item:
            reward = 20
            state.carry_item = 1
            state.exploration_rate = 1.0
        # If the agent is at the target location and carrying the item, reward it
        elif same_location(state.agent_location, self.target_location) and state.carry_item:
            reward = 20
        # Otherwise, penalize the agent
        else:
            reward = -1

        # Update the agent's reward attribute
        state.reward = reward


class Agent:
    def __init__(self, env: Environment, reward=0, learning_rate=0.1, discount_factor=0.9, exploration_rate=1,
                 exploration_decay=0.999, min_exploration_rate=0.05, statespace_size=statespace_size_g):
        """
        Initialize the agent.

        :param env: Reference to the environment the agent will operate in.
        :param reward: Initial reward for the agent.
        :param learning_rate: Learning rate for the agent's Q-learning.
        :param discount_factor: Discount factor for future rewards.
        :param exploration_rate: Initial exploration rate for the agent's epsilon-greedy policy.
        :param exploration_decay: Rate at which the exploration rate will decay over time.
        :param min_exploration_rate: Minimum exploration rate value.
        :param statespace_size: The size of the state space.
        """
        self.agent_location = env.agent_location
        self.carry_item = 0
        self.reward = reward
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.statespace_size = statespace_size
        self.loss_fn = torch.nn.MSELoss()
        self._prepare_torch()

    def _prepare_torch(self):
        """
        Setup the neural network models for Q-learning.
        """
        l1 = self.statespace_size
        l2 = l4_g
        l3 = l5_g
        # l4 = l4_g
        # l5 = l5_g
        # l6 = l6_g
        l7 = 4
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            # torch.nn.Linear(l3, l4),
            # torch.nn.ReLU(),
            # torch.nn.Linear(l4, l5),
            # torch.nn.ReLU(),
            # torch.nn.Linear(l5, l6),
            # torch.nn.ReLU(),
            torch.nn.Linear(l3, l7))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model2 = copy.deepcopy(self.model)
        self.model2.load_state_dict(self.model.state_dict())

    def prepare_state(self, state):
        """
        Convert the state into a format suitable for the neural network.

        :param state: The current state of the agent.
        :return: A reshaped numpy array representing the agent's state.
        """
        item_pos, agent_pos, target_pos, carry = state
        return np.concatenate([item_pos, agent_pos, target_pos, [carry]]).reshape(1, -1)

    def update_target(self):
        """
        Update the target model's weights with the main model's weights.
        """
        self.model2.load_state_dict(self.model.state_dict())

    def get_qvals(self, state):
        """
        Get Q-values for a given state.

        :param state: The state for which to get the Q-values.
        :return: Q-values for the state.
        """
        state1 = torch.from_numpy(state).float()
        qvals_torch = self.model(state1)
        qvals = qvals_torch.data.numpy()
        return qvals

    def get_maxQ(self, s):
        """
        Get the maximum Q-value for a given state.

        :param s: The state for which to get the maximum Q-value.
        :return: The maximum Q-value.
        """
        return torch.max(self.model2(torch.from_numpy(s).float())).float()

    def train_one_step(self, states, actions, targets, gamma=0.9):
        """
        Perform one step of Q-learning training.

        :param states: Batch of states.
        :param actions: Corresponding batch of actions.
        :param targets: Corresponding batch of target Q-values.
        :param gamma: Discount factor for future rewards.
        :return: Loss value for this training step.
        """
        state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
        action_batch = torch.Tensor(actions)
        Q1 = self.model(state1_batch)
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        Y = torch.tensor(targets)
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def move(self, action, env):
        """
        Update the agent's location based on the action taken.

        :param action: Action taken by the agent.
        :param env: Reference to the environment to check for boundaries and get reward.
        """
        # Movement logic for each action

        agent_location_xy = onehot_to_location(self.agent_location, env.width)

        # Movement logic for each action
        if action == 0 and agent_location_xy[0] > 0:
            agent_location_xy[0] -= 1
        elif action == 1 and agent_location_xy[0] < env.height - 1:
            agent_location_xy[0] += 1
        elif action == 2 and agent_location_xy[1] > 0:
            agent_location_xy[1] -= 1
        elif action == 3 and agent_location_xy[1] < env.width - 1:
            agent_location_xy[1] += 1

        self.agent_location = location_to_onehot(agent_location_xy, env.height, env.width)
        env.get_reward(self)

    def choose_action(self, env):
        """
        Choose an action based on the epsilon-greedy policy.

        :param env: Reference to the environment to prepare the state.
        :return: The chosen action.
        """
        state = self.prepare_state((env.item_location, self.agent_location, env.target_location, self.carry_item))
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(4)
        else:
            qvals = self.get_qvals(state)
            return np.argmax(qvals)

    def get_shortest_path(self, env, max_steps=200):
        """
        Find the shortest path to the target based on the agent's current Q-values.

        :param env: Reference to the environment.
        :param max_steps: Maximum number of steps to take.
        :return: A list of positions representing the shortest path.
        """
        agent_pos = self.agent_location
        item_pos = env.item_location
        target_pos = env.target_location
        done = False
        step_count = 0
        self.carry_item = 0
        shortest_path = [(onehot_to_location(self.agent_location, env.width)[0], onehot_to_location(self.agent_location, env.width)[1])]
        carry = 0
        while not done:
            state = self.prepare_state((item_pos, agent_pos, target_pos, carry))
            q_values = self.model2(torch.from_numpy(state).float())
            action = np.argmax(q_values.detach().numpy())
            self.move(action, env)

            agent_pos = self.agent_location
            carry = self.carry_item
            shortest_path.append((onehot_to_location(agent_pos, env.width)[0],
                                  onehot_to_location(agent_pos, env.width)[1]))
            if same_location(agent_pos, target_pos) and carry == 1:
                done = True
            step_count += 1
            if step_count >= max_steps:
                shortest_path = [(0, 0)] * max_steps
                break
        return shortest_path

def manhattan_distance(location1, location2):
    """
    :param location1: A tuple or list representing the coordinates (x, y) of the first location.
    :param location2: A tuple or list representing the coordinates (x, y) of the second location.
    :return: Integer representing the Manhattan distance between the two locations.
    :functionality: Calculates and returns the Manhattan distance between two points in a 2D grid.
    """
    return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])


def accuracy_test(env: Environment, agent: Agent, test_num=100):
    """
    :param env: An Environment object representing the environment in which the agent operates.
    :param agent: An Agent object representing the agent being tested.
    :param test_num: Integer representing the number of tests to perform.
    :return: Float representing the ratio of times the agent successfully found the exact shortest path
             in the given environment over the total number of tests.
    :functionality: Tests the agent's ability to match the exact shortest path in the given environment.
                    The accuracy is computed based on the radio of the number of times the agent's path
                    length matches the actual minimum distance to the total number of the test.
    """

    accuracy = 0

    for _ in range(test_num):
        # Regenerate random locations for items and agents in the environment
        env.regenerate_locations()
        # Calculate the actual minimum distance needed to pick the item and reach the target
        agent_location_xy = onehot_to_location(agent.agent_location, env.width)
        item_location_xy = onehot_to_location(env.item_location, env.width)
        target_location_xy = onehot_to_location(env.target_location, env.width)

        # Calculate the actual minimum distance needed to pick the item and reach the target
        actual_length = manhattan_distance(agent_location_xy, item_location_xy) + manhattan_distance(
            item_location_xy, target_location_xy) + 1

        # Initialize agent's state
        agent.agent_location = env.agent_location
        agent.carry_item = 0
        agent.reward = 0

        # Get the path length found by the agent
        path_length = len(agent.get_shortest_path(env))

        # Check if the path length found by the agent matches the actual minimum distance
        if path_length == actual_length:
            accuracy += 1

    # Calculate and return the accuracy ratio
    return accuracy / test_num


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self):
        return len(self.memory) >= self.capacity / 2


def train(env: Environment, agent: Agent, episodes=500, batch_size=400, update_fre=5):
    """
    Train the agent in the given environment for a specified number of episodes using experience replay.

    :param env: An Environment object representing the environment in which the agent operates.
    :param agent: An Agent object representing the agent being trained.
    :param episodes: Number of training episodes.
    :param batch_size: Size of the batch to sample from the experience replay.
    :param update_fre: Frequency of updates for the target model and calculating metrics.
    :return: Two lists - accuracies contains the agent's accuracy per update frequency,
             and losses_avg contains average loss per update frequency.
    """

    accuracies = []  # List to store accuracy measurements per update frequency
    losses = []  # List to store losses for each training step
    losses_avg = []  # List to store average losses per update frequency
    episode_values=[]

    memory = ExperienceReplay(30000)  # Initialize experience replay memory

    for episode in range(episodes):

        # Reset environment and agent's state
        env.regenerate_locations()
        agent.agent_location = env.agent_location
        agent.carry_item = 0
        agent.reward = 0
        count = 0
        done = 0

        # Continue the episode until the agent delivers the item or exceeds 500 steps
        while not (same_location(agent.agent_location, env.target_location) and agent.carry_item == 1 or count >= 500):

            # Agent chooses an action and updates its location and state
            action = agent.choose_action(env)
            old_state = agent.prepare_state((env.item_location, agent.agent_location, env.target_location, agent.carry_item)) + np.random.rand(1, statespace_size_g) / 20
            agent.move(action, env)
            new_state = agent.prepare_state((env.item_location, agent.agent_location, env.target_location, agent.carry_item)) + np.random.rand(1, statespace_size_g) / 20

            # Decay exploration rate
            agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * agent.exploration_decay)

            # Check if task is completed
            if agent.agent_location[0] == env.target_location[0] and agent.agent_location[1] == env.target_location[1] and agent.carry_item == 1:
                done = 1

            # Store the experience in the replay buffer
            memory.store((old_state, action, agent.reward, new_state, done))

            # Sample from the replay buffer and train the agent
            if memory.can_sample():
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states, d = zip(*experiences)

                # Calculate the target Q-values
                future_optimal_values = [agent.get_maxQ(next_state).item() for next_state in next_states]
                td_targets = [reward + agent.discount_factor * future_opt_value * (1 - d) for reward, future_opt_value, d in zip(rewards, future_optimal_values, d)]

                # Train the agent and store the loss
                loss = agent.train_one_step(states, actions, td_targets)
                losses.append(loss)

            count += 1

        # Update the target network and store metrics every `update_fre` episodes
        if len(losses)>0 and (episode + 1) % update_fre == 0:
            agent.update_target()
        if len(losses)>0:
            avg_loss = sum(losses[-update_fre:]) / update_fre
            losses_avg.append(avg_loss)
            avg_accuracy = accuracy_test(env, agent)
            accuracies.append(avg_accuracy)
            episode_values.append(episode)

        # Store metrics for the last episode as well
        if episode == episodes - 1:
            avg_loss = sum(losses[-update_fre:]) / update_fre
            losses_avg.append(avg_loss)
            agent.update_target()
            avg_accuracy = accuracy_test(env, agent)
            accuracies.append(avg_accuracy)
            episode_values.append(episode)

        # Print episode metrics
        if len(losses) > 0 and len(accuracies) > 0:
            print(f"Episode: {episode}, Losses Length: {len(losses)}, The loss over the last episodes: {sum(losses[-count:]) / count}, Test accuracy: {accuracies[-1]}")
        else:
            print(f"ExperienceReplay buffer storing...")

    return accuracies, losses, episode_values


def QL_test(env: Environment, agent: Agent, test_num=10000):
    """
    :param env: An Environment object representing the environment in which the agent operates.
    :param agent: An Agent object representing the agent being tested.
    :param test_num: Integer representing the number of test runs.
    :return: None
    :functionality: Tests the trained agent in the given environment for a specified number of runs.
                    The function prints out the details of each test (actual path length, agent's path length,
                    if they match, and the path itself) and the overall success rate at the end.
    """

    cnt = 0  # Counter for the number of successful paths found by the agent

    for i in range(test_num):
        # Regenerate the positions for the item and agent within the environment
        env.regenerate_locations()
        # Convert one_hot encoded locations back to 2D coordinates for Manhattan distance calculation
        agent_location_xy = onehot_to_location(agent.agent_location, env.width)
        item_location_xy = onehot_to_location(env.item_location, env.width)
        target_location_xy = onehot_to_location(env.target_location, env.width)

        # Calculate the optimal path length by summing the Manhattan distances
        actual_length = manhattan_distance(agent_location_xy, item_location_xy) + \
                        manhattan_distance(item_location_xy, target_location_xy) + 1

        agent.agent_location = env.agent_location  # Initialize the agent's location
        agent.carry_item = 0  # Reset the carry_item flag
        agent.reward = 0  # Reset the agent's reward

        # Get the shortest path according to the agent's learned policy
        path = agent.get_shortest_path(env)

        # Print details of the test
        print(actual_length, len(path), actual_length == len(path), path)

        # Check if the agent's path length matches the actual shortest path length
        if actual_length == len(path):
            cnt += 1  # Increment the counter if the path is correct

    # Print the success rate
    print('correct:', cnt, 'times out of', test_num, 'runs, correction rate =', str(100 * (cnt / test_num)) + '%')


def visualization(accuracies, losses, episodes):
    """
    :param accuracies: A list of accuracies measured during training episodes.
    :param losses: A list of losses recorded during training.
    :return: None
    :functionality: Generates two plots side-by-side, one for accuracies obtained during the episodes
                    and another for the losses encountered during each training step.
                    The purpose is to provide a visual representation of the agent's training progress.
    """

    plt.figure(figsize=(16, 6))  # Initialize the figure with dimensions 16x6
    # First subplot: Accuracies
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(episodes, accuracies, label='Accuracy')  # Plotting the accuracies data
    plt.xlabel('Episode')  # Label for the x-axis, representing every 5 episodes
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.title('Accuracy per Episode')  # Title for the first subplot
    plt.legend()  # Display the legend for the accuracies
    plt.grid(True)  # Enable the grid lines for better readability

    # Second subplot: Losses
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(list(range(len(losses))), losses, label='Loss')  # Plotting the losses data
    plt.xlabel('Step')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.title('Loss per Step')  # Title for the second subplot
    plt.legend()  # Display the legend for the losses
    plt.grid(True)  # Enable the grid lines for better readability

    plt.tight_layout()  # Adjust the layout to prevent overlap between plots
    plt.show()  # Display the plot


environment = Environment(4, 4)
q_agent = Agent(environment)
accuracies, losses, episodes = train(environment, q_agent)

# QL_test(environment, q_agent)
#
# visualization(accuracies, losses, episodes)