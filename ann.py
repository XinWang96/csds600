import numpy as np


def init_weight():
    w1 = [[0.1, 0, 0.3], [-0.2, 0.2, -0.4]]
    w2 = [[-0.4, 0.2], [0.1, -0.1], [0.6, -0.2]]
    b1 = [0.1, 0.2, 0.5]
    b2 = [-0.1, 0.6]

    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)

    weight = {"w1": w1, "w2": w2, "b1": b1, "b2": b2}

    return weight


def sigmoid(tensor):
    return 1 / (1 + np.exp(-tensor))


def d_sigmoid(tensor):
    return sigmoid(tensor) * (1 - sigmoid(tensor))


def epoch(input, weight, label, lr=0.1):
    res1 = np.dot(input, weight['w1']) + weight['b1']
    res2 = sigmoid(res1)

    res3 = np.dot(res2, weight['w2']) + weight['b2']
    res4 = sigmoid(res3)

    error = label - res4

    loss = np.sum((0.5 * (np.power((res4 - label), 2))))

    print(loss)

    tmp = d_sigmoid(res3)

    weight_update2 = np.dot(res2.T, error * tmp)

    bias_update2 = np.sum(error * tmp)

    tmp1 = np.dot(error * tmp, weight['w2'].T)

    delta_prime = d_sigmoid(res1)

    weight_update1 = np.dot(input.T, delta_prime * tmp1)
    bias_update1 = np.sum(delta_prime * tmp1)

    weight['w1'] += lr * weight_update1
    weight['b1'] += lr * bias_update1

    weight['w2'] += lr * weight_update2
    weight['b2'] += lr * bias_update2

    return weight


input = [[0.6, 0.1], [0.2, 0.3]]

label = [[1, 0], [0, 1]]

input = np.asarray(input)
label = np.asarray(label)

weight = init_weight()

#epoch(input,weight,label)

for i in range(10000):
    epoch(input, weight, label)