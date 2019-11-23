import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
import math

# load data set with selected or extracted features, features are discrete
# features are the columns after reward column
def generate_MDP_input(filename):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_variables = ['student', 'priorTutorAction', 'reward']
    features = feature_name[start_Fidx: len(feature_name)]

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = ['PS', 'WE']
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns + 1, Ns + 1))
    expectR = np.zeros((Nx, Ns + 1, Ns + 1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list) - 1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1

        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

    return [start_states, A, expectR, distinct_acts, distinct_states]


def calcuate_ECR(start_states, expectV):
        ECR_value = start_states.dot(np.array(expectV))
        return ECR_value

def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print(distinct_states)
    print(distinct_acts)
    print('Policy: ')
    print('state -> action, value-function')
    for s in range(Ns):
        print(distinct_states[s]+ " -> " + distinct_acts[vi.policy[s]] + ", "+str(vi.V[s]))


def calcuate_Q(T, R, V, gamma):
    q = np.zeros((T.shape[1], T.shape[0]))

    for s in range(T.shape[1]):
        for a in range(T.shape[0]):
            r = np.dot(R[a][s], T[a][s])
            Q = r + gamma * np.dot(T[a][s], V)
            q[s][a] = Q

    return q

def output_Qvalue(distinct_acts, distinct_states, Q):
    Ns = len(distinct_states)
    Na = len(distinct_acts)
    print('Q-value in Policy: ')
    print('state -> action, Q value function')
    for s in range(Ns):
        for a in range(Na):
            print(distinct_states[s] + " -> " + distinct_acts[a] + ", " + str(Q[s][a]))


def calculate_IS(filename, distinct_acts, distinct_states, Q, gamma, theta):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_variables = ['student', 'priorTutorAction', 'reward']
    features = feature_name[start_Fidx: len(feature_name)]

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1


    IS = 0
    random_prob = 0.5

    student_list = list(data['student'].unique())
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()


        cumul_policy_prob = 0
        cumul_random_prob = 0
        cumulative_reward = 0

        # calculate Importance Sampling Value for single student
        for i in range(1, len(row_list)):
            state = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            action = student_data.loc[row_list[i], 'priorTutorAction']
            reward = float(student_data.loc[row_list[i], 'reward'])

            Q_PS = Q[state][0]
            Q_WE = Q[state][1]

            #
            diff = Q_PS - Q_WE
            if diff > 60:
                diff = 60
            if diff < -60:
                diff = -60

            if action == 0:  # PS
                prob_logP = 1 / (1 + math.exp(-diff * theta))
            else:  # WE
                prob_logP = 1 / (1 + math.exp(diff * theta))


            cumul_policy_prob += math.log(prob_logP)
            cumul_random_prob += math.log(random_prob)
            cumulative_reward += math.pow(gamma, i-1) * reward
            i += 1

        weight = np.exp(cumul_policy_prob - cumul_random_prob)
        IS_reward = cumulative_reward * weight

        # cap the IS value
        if IS_reward > 300:
            IS_reward = 300
        if IS_reward < -300:
            IS_reward = -300
        IS += IS_reward

    IS = float(IS) / len(student_list)
    return IS

def induce_policy_MDP(filename, print_policy=False):
    # load data set with selected or extracted discrete features
    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(filename)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)
    vi.run()

    # output policy
    if print_policy:
        output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    # print('ECR value: ' + str(ECR_value))


    # calculate Q-value based on MDP
    Q = calcuate_Q(A, expectR, vi.V, 0.9)

    # output Q-value for each state-action pair
    # output_Qvalue(distinct_acts, distinct_states, Q)

    # evaluate policy using Importance Sampling
    IS_value = calculate_IS(filename, distinct_acts, distinct_states, Q, 0.9, 0.1)
    # print('IS value: ' + str(IS_value))
    
    if print_policy:
        print('ECR value: ' + str(ECR_value))
        print('IS value: ' + str(IS_value))
    
    return ECR_value, IS_value


if __name__ == "__main__":
    # extract filename from command
    parser = argparse.ArgumentParser()
    parser.add_argument("-input")
    args = parser.parse_args()
    filename = args.input
    induce_policy_MDP(filename, print_policy=True)
