# this file will define the learner class, along with required methods -
# we are taking inspiration (and in some cases borrowing heavily) from the following
# tutorial: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.utils import to_categorical
from collections import deque
from .evol_game import evol_env, evol_env_wf
from .dpsolve import backwards_induction, dp_env
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm


# Function to set hyperparameters for the learner - just edit this any time you
# want to screw around with them.
# or edit directly

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


# Run the function
make_keras_picklable()


class hyperparameters:
    '''
    class to store the hyperparemeters that control evoDM
    ...
    Args
    ------
    self: class hyperparameters

    Returns class hyperparameters
    '''

    def __init__(self):
        # Model training settings
        self.REPLAY_MEMORY_SIZE = 10000
        self.MASTER_MEMORY = True
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINIBATCH_SIZE = 100
        self.UPDATE_TARGET_EVERY = 310  # every 500 steps, update the target
        self.TRAIN_INPUT = "state_vector"
        self.DELAY = 0

        # Exploration settings
        self.DISCOUNT = 0.99
        self.epsilon = 1  # lowercase because its not a constant
        self.EPSILON_DECAY = 0.95
        self.MIN_EPSILON = 0.001
        self.LEARNING_RATE = 0.0001

        # settings control the evolutionary simulation
        self.NUM_EVOLS = 1  # how many evolutionary steps per time step
        self.SIGMA = 0.5
        self.NORMALIZE_DRUGS = True  # should fitness values for all landscapes be bound between 0 and 1?
        self.AVERAGE_OUTCOMES = False  # should we use the average of infinite evolutionary sims or use a single trajectory?
        self.DENSE = False  # will transition matrix be stored in dense or sparse format?
        # new evolutionary "game" every n steps or n *num_evols total evolutionary movements
        self.RESET_EVERY = 20
        self.EPISODES = 10  # FIXME needs to be 500, will change later
        self.N = 5
        self.RANDOM_START = False
        self.STARTING_GENOTYPE = 0  # default to starting at the wild type genotype
        self.NOISE = False  # should the sensor readings be noisy?
        self.NOISE_MODIFIER = 1  # enable us to increase or decrease the amount of noise in the system
        self.NUM_DRUGS = 4
        self.MIRA = True
        self.TOTAL_RESISTANCE = False
        self.PHENOM = 0
        # wright-fisher controls
        self.WF = False
        self.POP_SIZE = 10000
        self.GEN_PER_STEP = 1
        self.MUTATION_RATE = 1e-5

        # define victory conditions for player and pop
        self.PLAYER_WCUTOFF = 0.001
        self.POP_WCUTOFF = 0.999

        # define victory threshold
        self.WIN_THRESHOLD = 1000  # number of player actions before the game is called
        self.WIN_REWARD = 0

        # stats settings -
        self.AGGREGATE_STATS_EVERY = 1  # agg every episode


# This is the class for the learning agent
class DrugSelector:

    def __init__(self, hp, drugs=None):
        '''
        Initialize the DrugSelector class
        ...
        Args
        ------
        self: class DrugSelector
        hp: class hyperparameters
            hyperparameters that control the evodm architecture and the
            evolutionary simulations used to train it
        drugs: list of numeric matrices
            optional parameter - can pass in a list of drugs to use as the available actions.
            If not provided, drugs will be procedurally generated


        Returns class DrugSelector
        '''
        # hp stands for hyperparameters
        self.hp = hp
        if self.hp.WF:
            self.env = evol_env_wf(train_input=self.hp.TRAIN_INPUT,
                                   pop_size=self.hp.POP_SIZE,
                                   gen_per_step=self.hp.GEN_PER_STEP,
                                   mutation_rate=self.hp.MUTATION_RATE)
        else:
            # initialize the environment
            self.env = evol_env(num_evols=self.hp.NUM_EVOLS, N=self.hp.N,
                                train_input=self.hp.TRAIN_INPUT,
                                random_start=self.hp.RANDOM_START,
                                num_drugs=self.hp.NUM_DRUGS,
                                sigma=self.hp.SIGMA,
                                normalize_drugs=self.hp.NORMALIZE_DRUGS,
                                win_threshold=self.hp.WIN_THRESHOLD,
                                player_wcutoff=self.hp.PLAYER_WCUTOFF,
                                pop_wcutoff=self.hp.POP_WCUTOFF,
                                win_reward=self.hp.WIN_REWARD,
                                drugs=drugs,
                                add_noise=self.hp.NOISE,
                                noise_modifier=self.hp.NOISE_MODIFIER,
                                average_outcomes=self.hp.AVERAGE_OUTCOMES,
                                starting_genotype=self.hp.STARTING_GENOTYPE,
                                total_resistance=self.hp.TOTAL_RESISTANCE,
                                dense=self.hp.DENSE,
                                delay=self.hp.DELAY,
                                phenom=self.hp.PHENOM)

        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.hp.REPLAY_MEMORY_SIZE)
        self.master_memory = []
        self.target_update_counter = 0
        self.policies = []

    def create_model(self):

        model = Sequential()
        # need to change padding settings if using fitness to train model
        # because sequence may not be long enough
        if self.hp.TRAIN_INPUT == "state_vector":
            model.add(Conv1D(64, 3, activation="relu",
                             input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif self.hp.TRAIN_INPUT == "fitness":
            # have to change the kernel size because of the weird difference in environment shape
            model.add(Dense(64, activation="relu",
                            input_shape=self.env.ENVIRONMENT_SHAPE))
        elif self.hp.TRAIN_INPUT == "pop_size":
            model.add(Conv1D(64, 3, activation="relu",
                             input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        else:
            print(
                "please specify either state_vector, fitness, or pop_size for train_input when initializing the environment")
            return
        model.add(Dropout(0.2))
        model.add(Dense(28, activation="relu"))
        model.add(Dense(len(self.env.ACTIONS), activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.hp.LEARNING_RATE), metrics=['accuracy'])
        return model

    def get_current_state(self):
        """Get the current state based on the training input type"""
        if self.hp.TRAIN_INPUT == "state_vector":
            return np.array(self.env.state_vector).flatten()
        elif self.hp.TRAIN_INPUT == "fitness":
            return np.array(self.env.sensor[3]) if hasattr(self.env, 'sensor') and len(
                self.env.sensor) > 3 else np.array(self.env.fitness)
        elif self.hp.TRAIN_INPUT == "pop_size":
            return np.array(self.env.pop_size).flatten()
        else:
            return np.array(self.env.state_vector).flatten()

    def update_replay_memory(self, transition):
        """
        Update replay memory with a complete transition
        transition should be (current_state, action, reward, new_state, done)
        """
        if self.env.action_number > 1 + self.hp.DELAY:
            self.replay_memory.append(transition)

            # Update master memory - for diagnostic purposes only
            if self.hp.MASTER_MEMORY:
                if self.hp.TRAIN_INPUT == "fitness":
                    # Want to save the state vector history somewhere, regardless of what we use for training
                    self.master_memory.append([
                        self.env.episode_number,
                        self.env.action_number,
                        transition,
                        self.env.state_vector,
                        self.env.fitness
                    ])
                else:
                    self.master_memory.append([
                        self.env.episode_number,
                        self.env.action_number,
                        transition,
                        self.env.fitness
                    ])

    def train(self):
        """Train the main network"""
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.hp.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.hp.MINIBATCH_SIZE)

        # Get the current states
        current_states, new_current_states = self.get_current_states(minibatch=minibatch)

        current_qs_list = self.model.predict(current_states, verbose=0)
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        # Now we need to enumerate our batches
        X, y = self.enumerate_batch(minibatch=minibatch, future_qs_list=future_qs_list,
                                    current_qs_list=current_qs_list)

        self.model.fit(X, y, batch_size=self.hp.MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=None)

        # Update target update counter
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.hp.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def enumerate_batch(self, minibatch, future_qs_list, current_qs_list):
        """Function to enumerate batch and generate X/y for training -- this is where q table is updated"""
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            #THIS IS THE BELLMAN OPTIMALITY EQUATION
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.hp.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index].copy()
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Convert to numpy arrays and reshape
        if self.hp.TRAIN_INPUT == "state_vector":
            X = np.array(X).reshape(self.hp.MINIBATCH_SIZE, self.env.ENVIRONMENT_SHAPE[0],
                                    self.env.ENVIRONMENT_SHAPE[1])
        else:
            X = np.array(X).reshape(self.hp.MINIBATCH_SIZE, self.env.ENVIRONMENT_SHAPE[0])
        y = np.array(y)

        return X, y

    def get_current_states(self, minibatch):
        """Get current states from minibatch for training"""
        # Get current states from minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # Get future states from minibatch
        new_current_states = np.array([transition[3] for transition in minibatch])

        # Reshape to match expected input dimensions
        if self.hp.TRAIN_INPUT == "state_vector":
            current_states = current_states.reshape(self.hp.MINIBATCH_SIZE,
                                                    self.env.ENVIRONMENT_SHAPE[0],
                                                    self.env.ENVIRONMENT_SHAPE[1])
            new_current_states = new_current_states.reshape(self.hp.MINIBATCH_SIZE,
                                                            self.env.ENVIRONMENT_SHAPE[0],
                                                            self.env.ENVIRONMENT_SHAPE[1])
        else:
            current_states = current_states.reshape(self.hp.MINIBATCH_SIZE,
                                                    self.env.ENVIRONMENT_SHAPE[0])
            new_current_states = new_current_states.reshape(self.hp.MINIBATCH_SIZE,
                                                            self.env.ENVIRONMENT_SHAPE[0])

        return current_states, new_current_states

    def compute_implied_policy(self, update):
        '''
        Function to compute the implied policy learned by the DQ learner.
        ...
        Args
        ------
        self: class DrugSelector
        update: bool
            should we update the list of implied policies?

        Returns numeric matrix
            numeric matrix encodes policy in the same way as compute_optimal__policy
        '''
        policy = []

        if self.hp.TRAIN_INPUT == "state_vector":
            for s in range(len(self.env.state_vector)):
                # Create a one-hot encoded state
                temp_state_vector = np.zeros((2 ** self.env.N, 1))
                temp_state_vector[s] = 1

                # Temporarily set the environment state
                original_state = self.env.state_vector.copy()
                self.env.state_vector = temp_state_vector

                # Get the best action for this state
                action = np.argmax(self.get_qs())
                policy.append(to_categorical(action, num_classes=len(self.env.ACTIONS)))

                # Restore original state
                self.env.state_vector = original_state

        else:  # if the train input was fitness
            # Put together action list
            a_list = to_categorical([i for i in range(len(self.env.ACTIONS))])
            a_list = np.ndarray.tolist(a_list)

            for s in range(len(self.env.state_vector)):
                state_vector = np.zeros((2 ** self.env.N, 1))
                state_vector[s] = 1
                a_out = []

                for a in range(len(a_list)):
                    if self.hp.WF:
                        fit = np.dot(list(self.env.drugs[a].values()), state_vector)[0]
                    else:
                        fit = np.dot(self.env.drugs[a], state_vector)[0]

                    a_vec = deepcopy(a_list)[a]
                    a_vec.append(fit)
                    a_vec = np.array(a_vec)

                    # Reshape to feed into the model
                    tens = a_vec.reshape(-1, *self.env.ENVIRONMENT_SHAPE)

                    # Find the optimal action
                    action_a = self.model.predict(tens, verbose=0)[0].argmax()
                    a_out.append(action_a)

                policy.append(a_out)

        if update:
            self.policies.append([policy, self.env.episode_number])
        else:
            return policy

    def get_qs(self):
        """Function to get q vector for a given state"""
        if self.hp.TRAIN_INPUT == "state_vector":
            state_vector = np.array(self.env.state_vector)
            tens = state_vector.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "fitness":
            sensor = np.array(self.env.sensor[3]) if hasattr(self.env, 'sensor') and len(
                self.env.sensor) > 3 else np.array(self.env.fitness)
            tens = sensor.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "pop_size":
            tens = self.env.pop_size.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        else:
            return "error in get_qs()"

        return self.model.predict(tens, verbose=0)[0]


def compute_optimal_policy(agent, discount_rate=0.99, num_steps=20):
    '''
    Function to compute optimal policy based on reinforcement learning problem defined by the class DrugSelector
    ...
    Args
    ------
    agent: class DrugSelector

    Returns numeric matrix
        encoding optimal actions a for all states s in S
    '''
    env = dp_env(N=agent.env.N, sigma=agent.env.sigma,
                 drugs=agent.env.drugs, num_drugs=len(agent.env.drugs),
                 phenom=agent.env.PHENOM)

    policy, V = backwards_induction(env=env, discount_rate=discount_rate, num_steps=num_steps)
    return policy, V


def compute_optimal_action(agent, policy, step, prev_action=False):
    '''
    Function to compute the optimal action based on a deterministic policy.
    ...
    Args
    ------
    agent: class DrugSelector
    policy: numeric matrix
        encoding optimal actions a for all states s in S

    Returns int
        corresponding to optimal action
    '''
    index = [i for i, j in enumerate(agent.env.state_vector) if j == 1.][0]

    if prev_action:
        if agent.hp.TRAIN_INPUT == "state_vector":
            action = np.argmax(policy[index])
        else:
            action = policy[index][int(agent.env.prev_action)]
    else:
        action = policy[index][step]

    return action


def practice(agent, naive=False, standard_practice=False,
             dp_solution=False, pre_trained=False, discount_rate=0.99,
             policy="none", prev_action=False, wf=False, train_freq=1,
             compute_implied_policy_bool=False, num_episodes=0):
    '''
    Function that iterates through simulations to train the agent. Also used to test general drug cycling policies as controls for evodm
    ...
    Args
    ------
    agent: class DrugSelector
    naive: bool
        should a naive drug cycling policy be used
    standard_practice: bool
        should a drug cycling policy approximating standard clinical practice be tested
    dp_solution: bool
        should a gold-standard optimal policy computed using backwards induction of an MDP be tested
    pre_trained: bool
        is the provided agent pre-trained? (i.e. should we be updating weights and biases each time step)
    prev_action: bool
        are we evaluating implied policies or actual DP policies?
    discount_rate: float
    policy: numeric matrix
        encoding optimal actions a for all states s in S, defaults to "none" -
        in which case logic defined by bools will dictate which policy is used.
        If a policy is provided, it will supercede all other options and be tested
    train_freq: int
        how many time steps should pass between training the model.

    Returns rewards, agent, policy
        reward vector, trained agent including master memory dictating what happened, and learned policy (if applicable)
    '''
    if num_episodes == 0:
        num_episodes = agent.hp.EPISODES

    if dp_solution and not wf:
        dp_policy, V = compute_optimal_policy(agent, discount_rate=discount_rate,
                                              num_steps=agent.hp.RESET_EVERY)

    # This is a bit of a hack - we are coopting the code that tests the dp solution to
    # test user-provided policies that use the same format
    if policy != "none":
        dp_policy = policy
        dp_solution = True

    # Every given number of episodes we are going to track the stats
    reward_list = []
    ep_rewards = []
    count = 1

    for episode in tqdm(range(1, num_episodes + 1), ascii=True, unit='episodes',
                        disable=True if any([dp_solution, naive, pre_trained]) else False):

        episode_reward = 0

        if pre_trained:
            agent.hp.epsilon = 0

        # Initialize variables for transition storage
        current_state = None
        action = None

        for i in range(agent.hp.RESET_EVERY + 1):
            if i == 0:
                agent.env.step()
                current_state = agent.get_current_state()
                continue

            i_fixed = i - 1

            # Store previous state and action
            prev_state = current_state
            prev_action = action

            # Choose action based on policy
            if np.random.random() > agent.hp.epsilon:
                if naive:
                    if standard_practice and not wf:
                        if np.mean(agent.env.fitness) > 0.9:
                            avail_actions = [a for a in agent.env.ACTIONS if a != agent.env.action]
                            action = random.sample(avail_actions, k=1)[0]
                        else:
                            action = agent.env.action
                    else:
                        if wf:
                            action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
                            agent.env.update_drug(action)
                        else:
                            action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
                            agent.env.action = action
                elif dp_solution:
                    action = compute_optimal_action(agent, dp_policy, step=i_fixed, prev_action=prev_action)
                    if wf:
                        agent.env.update_drug(action)
                    else:
                        agent.env.action = action
                else:
                    action = np.argmax(agent.get_qs())
                    if wf:
                        agent.env.update_drug(action)
                    else:
                        agent.env.action = action
            else:
                # Get random action
                if standard_practice and not wf:
                    if np.mean(agent.env.fitness) > 0.9:
                        avail_actions = [a for a in agent.env.ACTIONS if a != agent.env.action]
                        action = random.sample(avail_actions, k=1)[0]
                    else:
                        action = agent.env.action
                elif dp_solution:
                    action = compute_optimal_action(agent, dp_policy, step=i_fixed, prev_action=prev_action)
                    if wf:
                        agent.env.update_drug(action)
                    else:
                        agent.env.action = action
                elif wf:
                    action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
                    agent.env.update_drug(action)
                else:
                    action = random.choice(agent.env.ACTIONS)
                    agent.env.action = action

            # Take the step
            agent.env.step()

            # Get reward and new state
            reward = agent.env.sensor[2] if hasattr(agent.env, 'sensor') and len(agent.env.sensor) > 2 else 0
            new_state = agent.get_current_state()
            done = agent.env.done

            episode_reward += reward

            # Store transition if we have a previous state
            if prev_state is not None and prev_action is not None:
                transition = (prev_state, prev_action, reward, current_state, done)
                agent.update_replay_memory(transition)

            # Update current state for next iteration
            current_state = new_state

            # Train the agent
            if not any([dp_solution, naive, pre_trained]):
                if count % train_freq == 0 and len(agent.replay_memory) >= agent.hp.MIN_REPLAY_MEMORY_SIZE:
                    agent.train()

                    if train_freq > agent.hp.RESET_EVERY and compute_implied_policy_bool:
                        if not agent.hp.NUM_EVOLS > 1:
                            agent.compute_implied_policy(update=True)

            if done:
                break

            count += 1

        # Store episode reward
        ep_rewards.append(episode_reward)

        # Log stats every AGGREGATE_STATS_EVERY episodes
        if not episode % agent.hp.AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            reward_list.append([episode, average_reward, min_reward, max_reward])

            # Update the implied policy vector
            if not any([dp_solution, naive, pre_trained]):
                if not agent.hp.NUM_EVOLS > 1 and compute_implied_policy_bool:
                    if not train_freq > agent.hp.RESET_EVERY:
                        agent.compute_implied_policy(update=True)

            # Debug information
            if episode % 10 == 0:
                print(f"Episode {episode}: Avg Reward: {average_reward:.3f}, "
                      f"Epsilon: {agent.hp.epsilon:.3f}, "
                      f"Replay Memory: {len(agent.replay_memory)}")

        # Decay epsilon
        if not naive and not pre_trained:
            if agent.hp.epsilon > agent.hp.MIN_EPSILON:
                agent.hp.epsilon *= agent.hp.EPSILON_DECAY
                agent.hp.epsilon = max(agent.hp.MIN_EPSILON, agent.hp.epsilon)

        # Reset environment for next iteration
        agent.env.reset()

    # Return appropriate policy based on the type of run
    if dp_solution:
        return_policy = dp_policy
        V = V if 'V' in locals() else []
    elif naive or pre_trained:
        return_policy = []
        V = []
    elif compute_implied_policy_bool:
        return_policy = agent.compute_implied_policy(update=False)
        V = []
    else:
        return_policy = []
        V = []

    return reward_list, agent, return_policy, V