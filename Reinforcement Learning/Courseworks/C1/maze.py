import numpy as np
import random
import matplotlib.pyplot as plt


def get_CID():
    return "02547890"  # Return your CID (add 0 at the beginning to ensure it is 8 digits long)


def get_login():
    return "kza23"  # Return your short imperial login


SEED = 70  # Used for creating reproducibe plots

# ================================================
# >>>>>>> HELPER CLASSES
# ================================================


class GraphicsMaze(object):
    def __init__(
        self,
        shape,
        locations,
        default_reward,
        obstacle_locs,
        absorbing_locs,
        absorbing_rewards,
        absorbing,
    ):
        self.shape = shape
        self.locations = locations
        self.absorbing = absorbing

        # Walls
        self.walls = np.zeros(self.shape)
        for ob in obstacle_locs:
            self.walls[ob] = 20

        # Rewards
        self.rewarders = np.ones(self.shape) * default_reward
        for i, rew in enumerate(absorbing_locs):
            self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

        # Print the map to show it
        self.paint_maps()

    def paint_maps(self):
        """
        Print the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)
        plt.show()

    def paint_state(self, state):
        """
        Print one state on the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        states = np.zeros(self.shape)
        states[state] = 30
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders + states)
        plt.show()

    def draw_deterministic_policy(self, Policy):
        """
        Draw a deterministic policy
        input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
        for state, action in enumerate(Policy):
            if self.absorbing[
                0, state
            ]:  # If it is an absorbing state, don't plot any action
                continue
            arrows = [
                r"$\uparrow$",
                r"$\rightarrow$",
                r"$\downarrow$",
                r"$\leftarrow$",
            ]  # List of arrows corresponding to each possible action
            action_arrow = arrows[action]  # Take the corresponding action
            location = self.locations[state]  # Compute its location on graph
            plt.text(
                location[1], location[0], action_arrow, ha="center", va="center"
            )  # Place it on graph
        plt.show()

    def draw_policy(self, Policy):
        """
        Draw a policy (draw an arrow in the most probable direction)
        input: Policy {np.array} -- policy to draw as probability
        output: /
        """
        deterministic_policy = np.array(
            [np.argmax(Policy[row, :]) for row in range(Policy.shape[0])]
        )
        self.draw_deterministic_policy(deterministic_policy)

    def draw_value(self, Value):
        """
        Draw a policy value
        input: Value {np.array} -- policy values to draw
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
        for state, value in enumerate(Value):
            if self.absorbing[
                0, state
            ]:  # If it is an absorbing state, don't plot any value
                continue
            location = self.locations[state]  # Compute the value location on graph
            plt.text(
                location[1], location[0], round(value, 2), ha="center", va="center"
            )  # Place it on graph
        plt.show()

    def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple deterministic policies
        input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(20, 8))
        for subplot in range(len(Policies)):  # Go through all policies
            ax = plt.subplot(
                n_columns, n_lines, subplot + 1
            )  # Create a subplot for each policy
            ax.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
            for state, action in enumerate(Policies[subplot]):
                if self.absorbing[
                    0, state
                ]:  # If it is an absorbing state, don't plot any action
                    continue
                arrows = [
                    r"$\uparrow$",
                    r"$\rightarrow$",
                    r"$\downarrow$",
                    r"$\leftarrow$",
                ]  # List of arrows corresponding to each possible action
                action_arrow = arrows[action]  # Take the corresponding action
                location = self.locations[state]  # Compute its location on graph
                plt.text(
                    location[1], location[0], action_arrow, ha="center", va="center"
                )  # Place it on graph
            ax.title.set_text(
                title[subplot]
            )  # Set the title for the graph given as argument
        plt.show()

    def draw_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policies (draw an arrow in the most probable direction)
        input: Policy {np.array} -- array of policies to draw as probability
        output: /
        """
        deterministic_policies = np.array(
            [
                [np.argmax(Policy[row, :]) for row in range(Policy.shape[0])]
                for Policy in Policies
            ]
        )
        self.draw_deterministic_policy_grid(
            deterministic_policies, title, n_columns, n_lines
        )

    def draw_value_grid(self, Values, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policy values
        input: Values {np.array of np.array} -- array of policy values to draw
        output: /
        """
        plt.figure(figsize=(20, 8))
        for subplot in range(len(Values)):  # Go through all values
            ax = plt.subplot(
                n_columns, n_lines, subplot + 1
            )  # Create a subplot for each value
            ax.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
            for state, value in enumerate(Values[subplot]):
                if self.absorbing[
                    0, state
                ]:  # If it is an absorbing state, don't plot any value
                    continue
                location = self.locations[state]  # Compute the value location on graph
                plt.text(
                    location[1], location[0], round(value, 1), ha="center", va="center"
                )  # Place it on graph
            ax.title.set_text(
                title[subplot]
            )  # Set the title for the graoh given as argument

        plt.show()


class Maze(object):
    def __init__(self):
        """
        Maze initialisation.
        input: /
        output: /
        """

        y, z = map(int, get_CID()[-2:])

        self._prob_success = 0.8 + 0.02 * (9 - y)  # float
        self._gamma = 0.8 + 0.02 * y  # float
        self._goal = z % 4  # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

        # Build the maze
        self._build_maze()

    def _build_maze(self):
        """
        Maze initialisation.
        input: /
        output: /
        """

        # Properties of the maze
        self._shape = (13, 10)
        self._obstacle_locs = [
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 7),
            (1, 8),
            (1, 9),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 7),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 7),
            (4, 1),
            (4, 7),
            (5, 1),
            (5, 7),
            (6, 5),
            (6, 6),
            (6, 7),
            (8, 0),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (10, 0),
        ]  # Location of obstacles
        self._absorbing_locs = [
            (2, 0),
            (2, 9),
            (10, 1),
            (12, 9),
        ]  # Location of absorbing states
        self._absorbing_rewards = [
            (500 if (i == self._goal) else -50) for i in range(4)
        ]
        self._starting_locs = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
        ]  # Reward of absorbing states
        self._default_reward = -1  # Reward for each action performs in the environment
        self._max_t = 500  # Max number of steps in the environment

        # Actions
        self._action_size = 4
        self._direction_names = [
            "N",
            "E",
            "S",
            "W",
        ]  # Direction 0 is 'N', 1 is 'E' and so on

        # States
        self._locations = []
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                loc = (i, j)
                # Adding the state to locations if it is no obstacle
                if self._is_location(loc):
                    self._locations.append(loc)
        self._state_size = len(self._locations)

        # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
        self._neighbours = np.zeros((self._state_size, 4))

        for state in range(self._state_size):
            loc = self._get_loc_from_state(state)

            # North
            neighbour = (loc[0] - 1, loc[1])  # North neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][
                    self._direction_names.index("N")
                ] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index("N")] = state

            # East
            neighbour = (loc[0], loc[1] + 1)  # East neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][
                    self._direction_names.index("E")
                ] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index("E")] = state

            # South
            neighbour = (loc[0] + 1, loc[1])  # South neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][
                    self._direction_names.index("S")
                ] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index("S")] = state

            # West
            neighbour = (loc[0], loc[1] - 1)  # West neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][
                    self._direction_names.index("W")
                ] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index("W")] = state

        # Absorbing
        self._absorbing = np.zeros((1, self._state_size))
        for a in self._absorbing_locs:
            absorbing_state = self._get_state_from_loc(a)
            self._absorbing[0, absorbing_state] = 1

        # Transition matrix
        self._T = np.zeros(
            (self._state_size, self._state_size, self._action_size)
        )  # Empty matrix of domension S*S*A
        for action in range(self._action_size):
            for outcome in range(4):  # For each direction (N, E, S, W)
                # The agent has prob_success probability to go in the correct direction
                if action == outcome:
                    prob = 1 - 3.0 * (
                        (1.0 - self._prob_success) / 3.0
                    )  # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
                # Equal probability to go into one of the other directions
                else:
                    prob = (1.0 - self._prob_success) / 3.0

                # Write this probability in the transition matrix
                for prior_state in range(self._state_size):
                    # If absorbing state, probability of 0 to go to any other states
                    if not self._absorbing[0, prior_state]:
                        post_state = self._neighbours[
                            prior_state, outcome
                        ]  # Post state number
                        post_state = int(
                            post_state
                        )  # Transform in integer to avoid error
                        self._T[prior_state, post_state, action] += prob

        # Reward matrix
        self._R = np.ones(
            (self._state_size, self._state_size, self._action_size)
        )  # Matrix filled with 1
        self._R = self._default_reward * self._R  # Set default_reward everywhere
        for i in range(len(self._absorbing_rewards)):  # Set absorbing states rewards
            post_state = self._get_state_from_loc(self._absorbing_locs[i])
            self._R[:, post_state, :] = self._absorbing_rewards[i]

        # Creating the graphical Maze world
        self._graphics = GraphicsMaze(
            self._shape,
            self._locations,
            self._default_reward,
            self._obstacle_locs,
            self._absorbing_locs,
            self._absorbing_rewards,
            self._absorbing,
        )

        # Reset the environment
        self.reset()

    def _is_location(self, loc):
        """
        Is the location a valid state (not out of Maze and not an obstacle)
        input: loc {tuple} -- location of the state
        output: _ {bool} -- is the location a valid state
        """
        if (
            loc[0] < 0
            or loc[1] < 0
            or loc[0] > self._shape[0] - 1
            or loc[1] > self._shape[1] - 1
        ):
            return False
        elif loc in self._obstacle_locs:
            return False
        else:
            return True

    def _get_state_from_loc(self, loc):
        """
        Get the state number corresponding to a given location
        input: loc {tuple} -- location of the state
        output: index {int} -- corresponding state number
        """
        return self._locations.index(tuple(loc))

    def _get_loc_from_state(self, state):
        """
        Get the state number corresponding to a given location
        input: index {int} -- state number
        output: loc {tuple} -- corresponding location
        """
        return self._locations[state]

    # Getter functions used only for DP agents
    # You DO NOT NEED to modify them
    def get_T(self):
        return self._T

    def get_R(self):
        return self._R

    def get_absorbing(self):
        return self._absorbing

    # Getter functions used for DP, MC and TD agents
    # You DO NOT NEED to modify them
    def get_graphics(self):
        return self._graphics

    def get_action_size(self):
        return self._action_size

    def get_state_size(self):
        return self._state_size

    def get_gamma(self):
        return self._gamma

    # Functions used to perform episodes in the Maze environment
    def reset(self):
        """
        Reset the environment state to one of the possible starting states
        input: /
        output:
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """
        self._t = 0
        self._state = self._get_state_from_loc(
            self._starting_locs[random.randrange(len(self._starting_locs))]
        )
        self._reward = 0
        self._done = False
        return self._t, self._state, self._reward, self._done

    def step(self, action):
        """
        Perform an action in the environment
        input: action {int} -- action to perform
        output:
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """

        # If environment already finished, print an error
        if self._done or self._absorbing[0, self._state]:
            print("Please reset the environment")
            return self._t, self._state, self._reward, self._done

        # Drawing a random number used for probaility of next state
        probability_success = random.uniform(0, 1)

        # Look for the first possible next states (so get a reachable state even if probability_success = 0)
        new_state = 0
        while self._T[self._state, new_state, action] == 0:
            new_state += 1
        assert (
            self._T[self._state, new_state, action] != 0
        ), "Selected initial state should be probability 0, something might be wrong in the environment."

        # Find the first state for which probability of occurence matches the random value
        total_probability = self._T[self._state, new_state, action]
        while (total_probability < probability_success) and (
            new_state < self._state_size - 1
        ):
            new_state += 1
            total_probability += self._T[self._state, new_state, action]
        assert (
            self._T[self._state, new_state, action] != 0
        ), "Selected state should be probability 0, something might be wrong in the environment."

        # Setting new t, state, reward and done
        self._t += 1
        self._reward = self._R[self._state, new_state, action]
        self._done = self._absorbing[0, new_state] or self._t > self._max_t
        self._state = new_state
        return self._t, self._state, self._reward, self._done


# ================================================
# >>>>>>> AGENT IMPLEMENTATIONS
# ================================================


class DP_agent(object):
    def init(self, env, threshold):
        self.R = env.get_R()
        self.T = env.get_T()
        self.gamma = env.get_gamma()
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.threshold = threshold

        # Stats for plotting only
        self.total_epochs = 0
        self.policy_updates = 0
        self.losses = []

    def policy_evaluation(self, policy):
        # Initialisation
        delta = (
            2 * self.threshold
        )  # Ensure delta is bigger than the threshold to start the loop
        V = np.zeros(self.state_size)  # Initialise value function to 0
        epochs = 0

        while delta > self.threshold:
            epochs += 1
            V_prime = (
                policy * (self.T * (self.R + self.gamma * V[:, np.newaxis])).sum(1)
            ).sum(1)

            delta = np.abs(V_prime - V).max()
            V = V_prime
        return V, epochs

    def solve_policy_iteration(self):
        # Initialisation
        policy = np.zeros((self.state_size, self.action_size))  # Vector of 0
        policy[:, 0] = 1  # Initialise policy to choose action 1 systematically
        V = np.zeros(self.state_size)  # Initialise value function to 0
        policy_stable = False  # Condition to stop the main loop

        while not policy_stable:
            V, epochs = self.policy_evaluation(policy)

            # +1 for evaluating the value as well
            self.total_epochs += epochs + 1

            max_indices = (
                (self.T * (self.R + self.gamma * V[:, np.newaxis])).sum(1)
            ).argmax(1)

            new_policy = np.zeros((self.state_size, self.action_size))
            new_policy[np.arange(self.state_size), max_indices] = 1

            policy_stable = not (policy - new_policy).any()
            policy = new_policy

        return policy, V

    def solve_value_iteration(self):
        delta = (
            self.threshold * 2
        )  # Setting value of delta to go through the first breaking condition
        V = np.zeros(self.state_size)  # Initialise values at 0 for each state
        policy = np.zeros((self.state_size, self.action_size))  # Initialisation

        while delta > self.threshold:
            self.total_epochs += 1

            intermediate_values = np.sum(
                self.T * (self.R + self.gamma * V[:, np.newaxis]), 1
            )
            V_prime = np.max(intermediate_values, 1)

            policy_indices = np.argmax(intermediate_values, 1)
            new_policy = np.zeros((self.state_size, self.action_size))
            new_policy[np.arange(self.state_size), policy_indices] = 1
            self.policy_updates += 1 if np.any(policy != new_policy) else 0
            policy = new_policy

            delta = np.max(np.abs(V_prime - V))
            self.losses.append(delta)
            V = V_prime

        return policy, V

    def solve(self, env, method="value-iteration", threshold=1):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - V {np.array} -- Corresponding value function
        """

        self.init(env, threshold)

        if method == "policy-iteration":
            return self.solve_policy_iteration()
        if method == "value-iteration":
            return self.solve_value_iteration()


class MC_agent(object):
    def init(self, env, epsilon, alpha):
        self.max_episodes = 500
        self.episode_max_length = 500

        self.gamma = env.get_gamma()
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.epsilon = epsilon
        self.alpha = alpha

        # Used for plots only
        self.total_errors = []

    def generate_episode(self, env, policy):
        states = []
        rewards = []
        actions = []

        t, state, _, done = env.reset()
        while not done and t < self.episode_max_length:
            action = np.random.choice(np.arange(self.action_size), p=policy[state])
            t, next_state, next_reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(next_reward)
            state = next_state

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        return states, rewards, actions, rewards.sum()

    def epsilon_greedy_distribution(self, Q, epsilon):
        policy = np.full(
            (self.state_size, self.action_size),
            epsilon / self.action_size,
        )
        max_indices = Q.argmax(axis=-1)
        policy[np.arange(self.state_size), max_indices] += 1 - epsilon
        return policy

    def greedy_distribution(self, Q):
        policy = np.zeros_like(Q)
        policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1
        return policy

    def calculate_returns(self, rewards):
        mask = np.tril(np.ones((len(rewards), len(rewards))))

        powers = (
            np.arange(0, -len(rewards), -1) + np.arange(len(rewards)).reshape(-1, 1)
        ) * mask

        return np.sum(
            mask * self.gamma**powers * rewards[::-1],
            1,
        )[::-1]

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env, epsilon=0.1, alpha=0.1):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - values {list of np.array} -- List of successive value functions for each episode
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode
        """
        # Extract the variables from env
        self.init(env, epsilon, alpha)

        Q = np.zeros((self.state_size, self.action_size))
        total_rewards = []
        values = [np.zeros(self.action_size)]
        for _ in range(self.max_episodes):
            policy = self.epsilon_greedy_distribution(Q, self.epsilon)
            states, rewards, actions, episode_reward = self.generate_episode(
                env, policy
            )
            total_rewards.append(episode_reward)
            returns = self.calculate_returns(rewards)

            # Reversing the arrays so the first occurance overwrites the ones that
            # come afterwards
            states = states[::-1]
            actions = actions[::-1]
            returns = returns[::-1]

            Q[states, actions] += self.alpha * (returns - Q[states, actions])

            # MSE
            self.total_errors.append(np.mean((returns - Q[states, actions]) ** 2))

            values.append(Q.max(axis=-1))

        policy = self.greedy_distribution(Q)
        return policy, values, total_rewards


class TD_agent(object):
    def init(self, env, epsilon, alpha):
        self.max_episodes = 500
        self.max_episde_length = 500

        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.gamma = env.get_gamma()

        self.epsilon = epsilon
        self.alpha = alpha

        # Used for plots only
        self.total_errors = []

    def epsilon_greedy_distribution(self, Q, epsilon):
        policy = np.full(
            (self.state_size, self.action_size),
            epsilon / self.action_size,
        )
        max_indices = Q.argmax(axis=-1)
        policy[np.arange(self.state_size), max_indices] += 1 - epsilon
        return policy

    def greedy_distribution(self, Q):
        policy = np.zeros_like(Q)
        policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1
        return policy

    def get_action(self, policy, state):
        return np.random.choice(np.arange(self.action_size), p=policy[state])

    def get_max_action(self, Q, state):
        return np.argmax(Q[state])

    def solve_sarsa(self, env):
        Q = np.random.rand(self.state_size, self.action_size)
        V = np.zeros(self.action_size)
        policy = self.epsilon_greedy_distribution(Q, self.epsilon)
        values = [V]
        total_rewards = []

        for _ in range(self.max_episodes):
            t, current_state, _, done = env.reset()

            episode_reward = 0
            episode_squared_errors = []
            while not done and t < self.max_episde_length:
                current_action = self.get_action(policy, current_state)
                t, next_state, reward, done = env.step(current_action)
                next_action = self.get_action(policy, next_state)

                target = reward + self.gamma * Q[next_state, next_action]
                error = target - Q[current_state, current_action]
                Q[current_state, current_action] += self.alpha * error

                current_state = next_state
                current_action = next_action
                episode_reward += reward

                # SE
                episode_squared_errors.append(error**2)

            policy = self.epsilon_greedy_distribution(Q, self.epsilon)
            values.append(Q.max(axis=-1))
            total_rewards.append(episode_reward)
            # MSE
            self.total_errors.append(np.mean(episode_squared_errors))

        policy = self.greedy_distribution(Q)
        return policy, values, total_rewards

    def solve_q_learning(self, env):
        Q = np.random.rand(self.state_size, self.action_size)
        values = [np.zeros(self.action_size)]
        policy = self.epsilon_greedy_distribution(Q, self.epsilon)
        total_rewards = []

        for _ in range(self.max_episodes):
            t, current_state, _, done = env.reset()

            episode_reward = 0
            episode_squared_errors = []
            while not done and t < self.max_episde_length:
                current_action = self.get_action(policy, current_state)
                t, next_state, reward, done = env.step(current_action)
                max_action = self.get_max_action(Q, next_state)

                target = reward + self.gamma * Q[next_state, max_action]
                error = target - Q[current_state, current_action]
                Q[current_state, current_action] += self.alpha * error
                current_state = next_state
                episode_reward += reward

                # SE
                episode_squared_errors.append(error**2)

            policy = self.epsilon_greedy_distribution(Q, self.epsilon)
            values.append(Q.max(axis=-1))
            total_rewards.append(episode_reward)
            # MSE
            self.total_errors.append(np.mean(episode_squared_errors))

        policy = self.greedy_distribution(Q)
        return policy, values, total_rewards

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env, method="sarsa", epsilon=0.1, alpha=0.1):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - values {list of np.array} -- List of successive value functions for each episode
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode
        """

        self.init(env, epsilon, alpha)

        if method == "q-learning":
            return self.solve_q_learning(env)
        if method == "sarsa":
            return self.solve_sarsa(env)


# ================================================
# >>>>>>> PLOTTING UTILITIES
# ================================================

"""
The two classes mock Maze and GraphicsMaze just so it is easier to create
vectorized plots and adjust parameters. They do not affect the original
classes and behave exactly the same, excluding the overwritten functions.
"""


# Overwrite init so it doesn't show the empty maze
def graphics_maze_mock_init(
    self,
    shape,
    locations,
    default_reward,
    obstacle_locs,
    absorbing_locs,
    absorbing_rewards,
    absorbing,
):
    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
        self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
        self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10


## Overwrite functions so they don't show the plot so it can be saved as pdf
def draw_deterministic_policy(self, Policy):
    plt.figure(figsize=(15, 10))
    plt.imshow(self.walls + self.rewarders)
    for state, action in enumerate(Policy):
        if self.absorbing[0, state]:
            continue
        arrows = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
        action_arrow = arrows[action]
        location = self.locations[state]
        plt.text(
            location[1],
            location[0],
            action_arrow,
            ha="center",
            va="center",
            fontsize=30,  # Make font bigger so it appears better in the pdf
        )


def draw_value(self, Value):
    plt.figure(figsize=(15, 10))
    plt.imshow(self.walls + self.rewarders)
    for state, value in enumerate(Value):
        if self.absorbing[0, state]:
            continue
        location = self.locations[state]
        plt.text(location[1], location[0], round(value, 2), ha="center", va="center")


GraphicsMazeMock = type(
    "GraphicsMazeMock", (GraphicsMaze,), GraphicsMaze.__dict__.copy()
)
GraphicsMazeMock.__init__ = graphics_maze_mock_init
GraphicsMazeMock.draw_deterministic_policy = draw_deterministic_policy
GraphicsMazeMock.draw_value = draw_value


# Create a mock of the Maze class so the __init__ method
# can be overwritten to allow for adding parameters.
# Use GraphicsMazeMock
def maze_mock_init(self, prob=None, gamma=None):
    y, z = map(int, get_CID()[-2:])
    if not (prob is None or gamma is None):
        self._prob_success = prob
        self._gamma = gamma
    else:
        self._prob_success = 0.8 + 0.02 * (9 - y)  # float
        self._gamma = 0.8 + 0.02 * y  # float
    self._goal = z % 4
    self._build_maze()
    self._graphics = GraphicsMazeMock(
        self._shape,
        self._locations,
        self._default_reward,
        self._obstacle_locs,
        self._absorbing_locs,
        self._absorbing_rewards,
        self._absorbing,
    )


MazeMock = type("MazeMock", (Maze,), Maze.__dict__.copy())
MazeMock.__init__ = maze_mock_init

"""
Save vector graphics plots
"""


def save_plot(filename):
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)


# Decorator to execute plot with a specific seed
def plot(save=False):
    def wrapper(func):
        np.random.seed(SEED)
        random.seed(SEED)

        func()

        if save:
            save_plot(func.__name__)
        plt.show()

    return wrapper


def save_results(agent, plot_name):
    maze = MazeMock()
    policy, value = agent.solve(maze)[:2]

    # DP returns a single list of values
    # Other two return list of values for every episode
    if not isinstance(value[-1], np.number):
        value = value[-1]

    maze.get_graphics().draw_policy(policy)
    save_plot(f"{plot_name}_policy")
    maze.get_graphics().draw_value(value)
    save_plot(f"{plot_name}_value")


# ================================================
# >>>>>>> PLOTTING FOR REPORT
# ================================================

"""
In order to execute the plotting functions uncomment the decorator '@plot(save=False)'
which will run the function and show the plot. Set save=True to save the plot as a pdf.
"""

"""
Final results for each agent
"""


# @plot(save=False)
def final_results():
    dp_agent = DP_agent()
    save_results(dp_agent, "dp")
    mc_agent = MC_agent()
    save_results(mc_agent, "mc")
    td_agent = TD_agent()
    save_results(td_agent, "td")


"""
Question 1 - Dynamic Programming
"""

"""
Question 1.1.1. Algorithm Choice
"""


# @plot(save=False)
def dp_iteration_comparison():
    maze = Maze()

    ## Dynamic programming (POLICY)
    dp_agent = DP_agent()
    dp_agent.solve(maze, "policy-iteration")
    policy_iterations = dp_agent.total_epochs

    ## Dynamic programming (VALUE)
    dp_agent = DP_agent()
    dp_agent.solve(maze, "value-iteration")
    value_iterations = dp_agent.total_epochs

    ## PLOTTING
    labels = ["Policy iteration", "Value iteration"]
    values = [policy_iterations, value_iterations]
    colors = ["red", "blue"]

    bar_positions = [0.45, 0.57]  # Adjusted positions to move bars closer

    plt.figure(figsize=(3, 3))
    plt.bar(bar_positions, values, width=0.1, color=colors)

    # Setting the x-ticks labels to show 'Policy iteration' and 'Value iteration'
    plt.xticks(bar_positions, labels)

    # Drawing horizontal lines connecting the y-axis values to the top of the bars
    lengths = [0.45, 0.95]
    for value, color, length in zip(values, colors, lengths):
        plt.axhline(y=value, color=color, linestyle="--", xmax=length)

    # Set yticks
    plt.yticks([0, policy_iterations, value_iterations])
    plt.ylabel("# of iterations")


"""
Question 1.1.2. Parameter Choice
"""


# @plot(save=False)
def dp_parameter_choice():
    maze = Maze()

    epochs = []
    policy_updates = []
    thresholds = np.logspace(2, -1, num=50)

    for threshold in thresholds:
        dp_agent = DP_agent()
        dp_agent.solve(maze, "value-iteration", threshold)

        epochs.append(dp_agent.total_epochs)
        policy_updates.append(dp_agent.policy_updates)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Number of Iterations", color=color)
    ax1.plot(thresholds, epochs, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")
    ax1.grid(True, which="major", ls="--")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Policy Updates", color=color)
    ax2.plot(thresholds, policy_updates, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Number of Iterations and Policy Updates vs Threshold")


"""#### Question 1.3. Environment Params"""


# @plot(save=False)
def environment_parameters():
    probabilities = [0.1, 0.25, 0.5]
    gammas = [0.1, 0.98]

    for prob in probabilities:
        maze = MazeMock(prob, 0.98)
        dp_agent = DP_agent()
        dp_policy, dp_value = dp_agent.solve(maze)
        maze.get_graphics().draw_policy(dp_policy)
        save_plot(f"dp_policy_{prob=}")
        maze.get_graphics().draw_value(dp_value)
        save_plot(f"dp_policy_{prob=}_values")

    for gamma in gammas:
        maze = MazeMock(0.5, gamma)
        dp_agent = DP_agent()
        _, dp_value = dp_agent.solve(maze)
        maze.get_graphics().draw_value(dp_value)
        save_plot(f"dp_values_{gamma=}")


"""
Question 2 - Monte Carlo
"""

"""
Question 2.1. Parameter Choice
"""


# @plot(save=False)
def epsilon_comparison():
    maze = MazeMock()
    epsilons = [0.01, 0.1, 0.8]

    _, axs = plt.subplots(3, 1, figsize=(6, 8))

    window_size = 20
    x = range(1, 501)

    colors = ["b", "g", "y"]

    for idx, epsilon in enumerate(epsilons):
        mc_agent = MC_agent()
        mc_agent.solve(maze, epsilon=epsilon)

        errors = np.array(mc_agent.total_errors)

        # Original MSE plot
        axs[idx].plot(
            x, errors, label=f"MSE (epsilon={epsilon:.4f})", color=colors[idx]
        )

        # Compute running average using cumsum and a sliding window
        cumsum = np.cumsum(np.insert(errors, 0, 0))
        avg_errors = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        axs[idx].plot(
            x[window_size - 1 :],
            avg_errors,
            "r-",
            label=f"Running Avg (k={window_size})",
        )
        axs[idx].grid(True)

        # Turn off x-axis labels for the first two subplots
        if idx != 2:
            axs[idx].set_xticklabels([])
        else:
            axs[idx].set_xlabel("Episodes")

        axs[idx].legend(loc="upper right")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)


# @plot(save=False)
def alpha_comparison():
    maze = MazeMock()
    alphas = [0.01, 0.1, 0.9]

    _, axs = plt.subplots(3, 1, figsize=(6, 8))

    window_size = 20
    x = range(1, 501)

    colors = ["b", "g", "y"]

    for idx, alpha in enumerate(alphas):
        mc_agent = MC_agent()
        mc_agent.solve(maze, alpha=alpha)

        errors = np.array(mc_agent.total_errors)

        # Original MSE plot
        axs[idx].plot(x, errors, label=f"MSE (alpha={alpha:.4f})", color=colors[idx])

        # Compute running average using cumsum and a sliding window
        cumsum = np.cumsum(np.insert(errors, 0, 0))
        avg_errors = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        axs[idx].plot(
            x[window_size - 1 :],
            avg_errors,
            "r-",
            label=f"Running Avg (k={window_size})",
        )
        axs[idx].grid(True)

        # Turn off x-axis labels for the first two subplots
        if idx != 2:
            axs[idx].set_xticklabels([])
        else:
            axs[idx].set_xlabel("Episodes")

        axs[idx].legend(loc="upper right")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)


"""Question 2.3. Learning Curve"""


# @plot(save=False)
def mc_learning_curve():
    maze = MazeMock()
    rewards_per_run = []
    num_runs = 25
    num_episodes = 500

    for _ in range(num_runs):
        mc_agent = MC_agent()
        _, _, total_rewards = mc_agent.solve(maze)
        rewards_per_run.append(total_rewards)

    mean_rewards = np.mean(rewards_per_run, axis=0)
    std_rewards = np.std(rewards_per_run, axis=0)

    episode_numbers = range(1, num_episodes + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episode_numbers, mean_rewards, label="Mean Rewards", color="blue")
    plt.fill_between(
        episode_numbers,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="Standard Deviation",
        color="red",
    )
    plt.title("Learning Curve of the MC Agent")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Non-discounted Sum of Rewards")
    plt.legend()
    plt.grid()


"""
Question 3 - Temporal Difference Learning
"""


"""
Question 3.1. Algorithm choice
"""


# @plot(save=False)
def td_reward_comparison():
    maze = MazeMock()

    # Q-Learning
    td_agent = TD_agent()
    _, _, ql_total_rewards = td_agent.solve(maze, method="q-learning")

    # SARSA
    td_agent = TD_agent()
    _, _, sarsa_total_rewards = td_agent.solve(maze, method="sarsa")

    window_size = 20
    x = range(1, 501)

    cumsum_ql = np.cumsum(np.insert(ql_total_rewards, 0, 0))
    cumsum_sarsa = np.cumsum(np.insert(sarsa_total_rewards, 0, 0))

    avg_errors_ql = (cumsum_ql[window_size:] - cumsum_ql[:-window_size]) / window_size
    avg_errors_sarsa = (
        cumsum_sarsa[window_size:] - cumsum_sarsa[:-window_size]
    ) / window_size

    plt.plot(x[window_size - 1 :], avg_errors_ql, "b", label=f"Q-Learning")
    plt.plot(x[window_size - 1 :], avg_errors_sarsa, "r", label=f"SARSA")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Q-Learning vs SARSA (Running average over {window_size} rewards)")
    plt.legend(loc="lower right")
    plt.grid()


""" Question 3.3. Parameter Influence"""


# @plot(save=False)
def td_epsilon_comparison():
    maze = MazeMock()
    epsilons = [0.01, 0.1, 0.8]

    _, axs = plt.subplots(3, 1, figsize=(6, 8))

    window_size = 50

    colors = ["b", "g", "y"]

    for idx, epsilon in enumerate(epsilons):
        td_agent = TD_agent()
        td_agent.solve(maze, epsilon=epsilon)

        errors = np.array(td_agent.total_errors)

        x = range(1, td_agent.max_episodes + 1)

        # Original MSE plot
        axs[idx].plot(
            x, errors, label=f"MSE (epsilon={epsilon:.4f})", color=colors[idx]
        )

        # Compute running average using cumsum and a sliding window
        cumsum = np.cumsum(np.insert(errors, 0, 0))
        avg_errors = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        axs[idx].plot(
            x[window_size - 1 :],
            avg_errors,
            "r-",
            label=f"Running Avg (k={window_size})",
        )
        axs[idx].grid(True)

        # Turn off x-axis labels for the first two subplots
        if idx != 2:
            axs[idx].set_xticklabels([])
        else:
            axs[idx].set_xlabel("Episodes")

        axs[idx].legend(loc="upper right")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)


# @plot(save=False)
def td_alpha_comparison():
    maze = MazeMock()
    alphas = [0.01, 0.1, 0.9]

    _, axs = plt.subplots(3, 1, figsize=(6, 8))

    window_size = 20

    colors = ["b", "g", "y"]

    for idx, alpha in enumerate(alphas):
        td_agent = TD_agent()
        td_agent.solve(maze, alpha=alpha)

        errors = np.array(td_agent.total_errors)

        x = range(1, td_agent.max_episodes + 1)

        # Original MSE plot
        axs[idx].plot(x, errors, label=f"MSE (alpha={alpha:.4f})", color=colors[idx])

        # Compute running average using cumsum and a sliding window
        cumsum = np.cumsum(np.insert(errors, 0, 0))
        avg_errors = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        axs[idx].plot(
            x[window_size - 1 :],
            avg_errors,
            "r-",
            label=f"Running Avg (k={window_size})",
        )
        axs[idx].grid(True)

        # Turn off x-axis labels for the first two subplots
        if idx != 2:
            axs[idx].set_xticklabels([])
        else:
            axs[idx].set_xlabel("Episodes")

        axs[idx].legend(loc="upper right")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)


# ================================================
# >>>>>>> MAIN
# ================================================

if __name__ == "__main__":
    maze = Maze()

    # ## Question 1: Dynamic-Programming learning
    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)

    print("Results of the DP agent:\n")
    maze.get_graphics().draw_policy(dp_policy)
    maze.get_graphics().draw_value(dp_value)

    # Question 2: Monte-Carlo learning
    mc_agent = MC_agent()
    mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

    print("Results of the MC agent:\n")
    maze.get_graphics().draw_policy(mc_policy)
    maze.get_graphics().draw_value(mc_values[-1])

    ## Question 3: Temporal-Difference learning
    td_agent = TD_agent()
    td_policy, td_values, total_rewards = td_agent.solve(maze)

    print("Results of the TD agent:\n")
    maze.get_graphics().draw_policy(td_policy)
    maze.get_graphics().draw_value(td_values[-1])
