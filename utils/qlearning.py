import gym
import connect4_gym
import numpy as np
import random
import operator
import pickle


class QLearning:
    def __init__(self, q_table={}, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make('Connect4-v0')
        self.q_table = q_table
        self.state = self.get_current_state()
        self.valid_actions = self.get_available_moves()

    def get_current_state(self):
        return self.env.board.board

    def get_qvalue(self, state, action=None):
        key = str(state)
        if key in self.q_table.keys():
            if action is None:
                return self.q_table[key]
            if action in self.q_table[key].keys():
                return self.q_table[key][action]
        elif action is None:
            return {}
        return 0

    def get_best_move(self):
        pass

    def update_qvalue(self, state, action, value):
        key = str(state)
        if key in self.q_table.keys():
            self.q_table[key][action] = value
        else:
            self.q_table[key] = {action: value}

    def get_available_moves(self):
        valid_actions = []
        for col in range(self.state.shape[1]):
            if self.env.board.column_fill[col] != self.state.shape[0]:
                valid_actions.append(col)
        return valid_actions

    def train(self, n_episodes=1000000):
        """Training the agent"""
        for i in range(1, n_episodes):
            self.env.reset()
            state = self.state

            epochs, penalties, reward = 0, 0, 0
            done = False

            while not done:
                if len(self.get_available_moves()) == 0:
                    print("match draw!!")
                    break

                # player 1 : Agent
                if random.uniform(0, 1) < self.epsilon:
                    action = random.sample(self.get_available_moves(), 1)[0]
                    #action = random.sample(self.valid_actions, 1)[0]
                else:
                    pass
                    #actions = [action for action in self.get_qvalue(state) if action in self.valid_actions]
                    #action = max(self.get_qvalue(state).items(), key=operator.itemgetter(1))[0]
                    #if len(actions) is 0:
                    #    action = random.sample(self.valid_actions, 1)[0]
                    #else:
                    #    # need to alter it to select action for which q_value is max
                    #    action = max(actions)
                next_state, reward, done, info = self.env.step(action)
                assert info['last_player'] == 1
                next_state = self.env.board.board

                # update Q Table
                old_value = self.get_qvalue(state, action)
                try:
                    next_max = max(self.get_qvalue(next_state).items(), key=operator.itemgetter(1))[1]
                except ValueError as e:
                    next_max = 0
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.update_qvalue(state, action, new_value)

                if abs(reward) == 1:
                    break
                self.env.render()
                # player 2
                #ret, valid_move_list = self.get_available_moves()
                if len(self.get_available_moves()) is 0:
                    print("match draw!!")
                    break
                #p2_action = random.sample(self.valid_actions, 1)[0]
                p2_action = random.sample(self.get_available_moves(), 1)[0]
                next_state, reward, done, info = self.env.step(p2_action)
                self.env.render()
                epochs += 1
                if abs(reward) == 1:
                    break

            if i % 100 == 0:
                # clear_output(wait=True)
                print(f"Episode: {i}")
                print(len(self.q_table))

        print("Training finished.\n")
        print(len(self.q_table))
        with open('q_table.pickle', 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__=="__main__":
    agent = QLearning(epsilon=2)
    agent.train()