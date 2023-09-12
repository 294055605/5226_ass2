import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import random

# parameter
statespace_size_g=7# statespace_size 是模型输入层的神经元数量。如果 statespace_size=48，这意味着模型期望接收一个形状为 [batch_size, 48] 的张量（Tensor）作为输入。
# 这里的 batch_size 表示一批数据的数量。在单个数据点的情况下，batch_size 就是1。
l2_g= 256 # 150
l3_g = 128 # 200


def same_location(location1, location2):
    # Check if either location is None
    if location1 is None or location2 is None:
        return False

    # Compare the locations
    return location1[0] == location2[0] and location1[1] == location2[1]


class Environment:
    def __init__(self, height, width):
        self.item_location = None
        self.agent_location = None
        self.width = width
        self.height = height
        self.target_location = None
        self.regenerate_locations()

    def regenerate_locations(self):
        self.set_target_location(self.generate_random_location())
        self.set_agent_location(self.generate_random_location([self.target_location]))
        self.set_item_location(self.generate_random_location([self.target_location, self.agent_location]))

    def set_target_location(self, new_location):
        self.target_location = new_location

    def set_agent_location(self, new_location):
        self.agent_location = new_location

    def set_item_location(self, new_location):
        self.item_location = new_location

    def generate_random_location(self, exclude=()):
        def location_exist(new_location):
            for loc in exclude:
                if same_location(loc, new_location):
                    return True
            return False

        location = [np.random.randint(0, self.height), np.random.randint(0, self.width)]
        while location_exist(location):  # 意味着 如果经过上面函数的判断，一旦结果是TRUE ，就要再次生成一个点，用TRUE做while循环的开关，直到生成的新的location。
            location = [np.random.randint(0, self.height), np.random.randint(0, self.width)]
        return location

    def get_reward(self, state):
        if same_location(state.agent_location, self.item_location) and not state.carry_item:
            reward = 20
            state.carry_item = 1
            state.exploration_rate = 1.0
        elif same_location(state.agent_location, self.target_location) and state.carry_item:
            reward = 20
        else:
            reward = -1
        state.reward = reward

class Agent:
    def __init__(self, env: Environment, reward=0, learning_rate=0.01, discount_factor=0.9, exploration_rate=1,
                 exploration_decay=0.999, min_exploration_rate=0.05, statespace_size=statespace_size_g):
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
        l1 = self.statespace_size
        l2 = l2_g
        l3 = l3_g
        l4 = 4
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model2 = copy.deepcopy(self.model)
        self.model2.load_state_dict(self.model.state_dict())

    # 返回一个1*N的numpy数组# [1, 1, 2, 2, 3, 3, 0]
    def prepare_state(self, state):
        item_pos, agent_pos, target_pos, carry = state  # item_pos = [1,1] agent_pos = [2,2] carry = 0
        return np.array(item_pos + agent_pos + target_pos + [carry]).reshape(1, -1)  # [1 1 2 2 3 3 0]

    # 赋值model2权重
    def update_target(self):
        self.model2.load_state_dict(self.model.state_dict())

    # input: numpy数组 outputs： NumPy 数组
    def get_qvals(self, state):
        state1 = torch.from_numpy(state).float()
        qvals_torch = self.model(state1)
        qvals = qvals_torch.data.numpy()
        return qvals

    # 对给定的状态 s 使用模型 self.model2 来预测 Q 值，并返回最大的 Q 值
    # input:  NumPy 数组
    # output: PyTorch 张量（tensor），其中包含了在给定状态 s 下所有可能动作的最大 Q 值
    # max_q_value = agent.get_maxQ(example_state)
    # max_q_value_float = max_q_value.item()
    def get_maxQ(self, s):
        return torch.max(self.model2(torch.from_numpy(s).float())).float()

    # 更新 model 函数返回损失的标量值（Python 浮点数
    def train_one_step(self, states, actions, targets, gamma=0.9):
        # pass to this function: state1_batch, action_batch, TD_batch
        state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])# 将状态从 NumPy 数组转换为 PyTorch 张量，并进行拼接。
        action_batch = torch.Tensor(actions)# 将动作转换为 PyTorch 张量。
        Q1 = self.model(state1_batch)# 使用模型 self.model 预测给定状态下每个动作的 Q 值。
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()# 从预测的 Q 值中提取实际执行动作的 Q 值。
        Y = torch.tensor(targets)# 将目标 Q 值转换为 PyTorch 张量。
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def move(self, action, env):  # 返回state 给environment，详见 class environment中最后一个define
        if action == 0 and self.agent_location[0] > 0:
            self.agent_location[0] -= 1
        elif action == 1 and self.agent_location[0] < env.height - 1:
            self.agent_location[0] += 1
        elif action == 2 and self.agent_location[1] > 0:
            self.agent_location[1] -= 1
        elif action == 3 and self.agent_location[1] < env.width - 1:
            self.agent_location[1] += 1
        env.get_reward(self)# self 指的是 Agent 类的一个实例

    def choose_action(self, env):
        state = self.prepare_state((env.item_location, self.agent_location, env.target_location, self.carry_item))
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(4)
        else:
            qvals = self.get_qvals(state)
            return np.argmax(qvals)

    def get_shortest_path(self, env, max_steps=200):
        agent_pos = self.agent_location
        item_pos = env.item_location
        target_pos = env.target_location
        done = False
        step_count = 0
        self.carry_item = 0
        shortest_path = [(agent_pos[0], agent_pos[1])]
        carry = 0
        while not done:
            state = self.prepare_state((item_pos, agent_pos, target_pos, carry))
            q_values = self.model2(torch.from_numpy(state).float())
            action = np.argmax(q_values.detach().numpy())
            self.move(action, env)

            agent_pos = self.agent_location
            carry = self.carry_item
            shortest_path.append((agent_pos[0], agent_pos[1]))
            if agent_pos[0] == env.target_location[0] and agent_pos[1] == env.target_location[1] and carry == 1:
                done = True
            step_count += 1
            if step_count >= max_steps:
                # print(f"Reached the maximum number of steps ({max_steps})!")
                shortest_path = [(0, 0)] * max_steps
                break
        return shortest_path


def manhattan_distance(location1, location2):
    return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])

def accuracy_test(env: Environment, agent: Agent, test_num=100):

    accuracy = 0

    for _ in range(test_num):
        env.regenerate_locations()
        actual_length = manhattan_distance(env.agent_location, env.item_location) + manhattan_distance(
            env.item_location, env.target_location) + 1
        agent.agent_location = env.agent_location
        agent.carry_item = 0
        agent.reward = 0
        path_length = len(agent.get_shortest_path(env))
        if path_length == actual_length:
            accuracy += 1

    return accuracy / test_num

    # total_accuracy += accuracy
    # return total_accuracy / test_num


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 这段代码的目的是在有限的内存中保存最近的经验。当新经验到来时，最旧的经验（如果缓冲区已满）会被替换掉
    def store(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity# 更新存储位置。如果达到缓冲区的容量，它会回到列表的开始并从头开始覆盖经验。
    # 随机取出batch个数据
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # 存储数据＞batch，返回True，否则False
    def can_sample(self):
        return len(self.memory) >= self.capacity / 2


def train(env: Environment, agent: Agent, episodes=500, batch_size=200, update_fre = 5):
    accuracies = []
    losses = []
    losses_avg = []

    memory = ExperienceReplay(10000)  # 初始化经验回放内存

    for episode in range(episodes):
        env.regenerate_locations()
        agent.agent_location = env.agent_location
        agent.carry_item = 0
        agent.reward = 0
        count = 0
        done = 0

        while not (agent.agent_location[0] == env.target_location[0] and
                   agent.agent_location[1] == env.target_location[1] and agent.carry_item == 1 or count >= 500):
            action = agent.choose_action(env)
            old_state = agent.prepare_state((env.item_location, agent.agent_location, env.target_location, agent.carry_item)) + np.random.rand(1,7) /20
            agent.move(action, env)
            new_state = agent.prepare_state((env.item_location, agent.agent_location, env.target_location, agent.carry_item)) + np.random.rand(1,7) /20
            agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * agent.exploration_decay)

            if agent.agent_location[0] == env.target_location[0] and agent.agent_location[1] == env.target_location[1] and agent.carry_item == 1:
                done = 1

            # 存储经验
            memory.store((old_state, action, agent.reward, new_state, done))

            # 经验回放
            if memory.can_sample():
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states, d = zip(*experiences)

                # 计算目标Q值
                future_optimal_values = [agent.get_maxQ(next_state).item() for next_state in next_states]
                # print(list(zip(rewards, future_optimal_values, d)))
                td_targets = [reward + agent.discount_factor * future_opt_value * (1-d) for reward, future_opt_value, d in zip(rewards, future_optimal_values, d)]

                # 训练网络
                loss = agent.train_one_step(states, actions, td_targets)
                losses.append(loss)  # 新增：将当前episode的loss添加到列表中

            count += 1
        # avg_accuracy = accuracy_test(env, agent)
        # accuracies.append(avg_accuracy)

        if (episode + 1) % update_fre == 0:
            avg_loss = sum(losses[-update_fre:]) / update_fre
            losses_avg.append(avg_loss)
            agent.update_target()
            avg_accuracy = accuracy_test(env, agent)
            accuracies.append(avg_accuracy)

        if episode == episodes - 1:
            avg_loss = sum(losses[-update_fre:]) / update_fre
            losses_avg.append(avg_loss)
            agent.update_target()
            avg_accuracy = accuracy_test(env, agent)
            accuracies.append(avg_accuracy)

        if len(losses) > 0 and len(accuracies) == 0:
            print(episode, len(losses), sum(losses[-count:]) / count)
        elif len(losses) == 0 and len(accuracies) > 0:
            print(episode, len(losses), accuracies[-1])
        elif len(losses) > 0 and len(accuracies) > 0:
            print(episode, len(losses), sum(losses[-count:]) / count, accuracies[-1])
        else:
            print(f"Episode {episode}: No losses recorded.")

    return accuracies, losses_avg


def QL_test(env: Environment, agent: Agent, test_num=10000):
    cnt = 0
    for i in range(test_num):
        environment.regenerate_locations()
        # print('start location: ', environment.agent_location)
        # print('item location', environment.item_location)
        # print('target location', environment.target_location)
        actual_length = manhattan_distance(environment.agent_location, environment.item_location) + manhattan_distance(
            environment.item_location, environment.target_location) + 1
        q_agent.agent_location = environment.agent_location
        q_agent.carry_item = 0
        q_agent.reward = 0
        path = q_agent.get_shortest_path(environment)
        print(actual_length, len(path), actual_length == len(path), path)
        if actual_length == len(path):
            cnt += 1
    print('correct:', cnt, 'times out of', test_num, 'runs, correction rate =', str(100 * (cnt / test_num)) + '%')


def visualization(accuracies, losses):
    plt.figure(figsize=(15, 6))

    # Accuracies
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Episode x 5')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Episode')
    plt.legend()
    plt.grid(True)

    # Losses
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss per Training Step')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    environment = Environment(4, 4)
    q_agent = Agent(environment)
    accuracies, losses = train(environment, q_agent)
    QL_test(environment, q_agent)
    visualization(accuracies, losses)

