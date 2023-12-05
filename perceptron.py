# Perceptron Algorithm for AND Logic Gate
w1, w2, b = 0.5, 0.5, -1

def activate(x):
    return 1 if x >= 0 else 0

def train_perceptron(inputs, desired_outputs, learning_rate, epochs):
    global w1, w2, b
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            target_output = desired_outputs[i]
            output = activate(w1 * A + w2 * B + b)
            error = target_output - output
            w1 += learning_rate * error * A
            w2 += learning_rate * error * B
            b += learning_rate * error
            total_error += abs(error)
        if total_error == 0:
            break


inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
desired_outputs = [0, 0, 0, 1]
learning_rate = 0.1
epochs = 100

train_perceptron(inputs, desired_outputs, learning_rate, epochs)

for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(w1 * A + w2 * B + b)
    print(f"Input: ({A}, {B})  Output: {output}")
print(f"{w1} {w2}")


#OR Gate

w1, w2, b = 0.5, 0.5, -1

def activate(x):
    return 1 if x >= 0 else 0

def train_perceptron(inputs, desired_outputs, learning_rate, epochs):
    global w1, w2, b
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            target_output = desired_outputs[i]
            output = activate(w1 * A + w2 * B + b)
            error = target_output - output
            w1 += learning_rate * error * A
            w2 += learning_rate * error * B
            b += learning_rate * error
            total_error += abs(error)
        if total_error == 0:
            break


inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
desired_outputs = [0, 1,1,1]
learning_rate = 0.1
epochs = 100

train_perceptron(inputs, desired_outputs, learning_rate, epochs)

for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(w1 * A + w2 * B + b)
    print(f"Input: ({A}, {B})  Output: {output}")
print(f"{w1} {w2}")


#XOR Gate
w1, w2, b = 0.5, 0.5, -1

def activate(x):
    return 1 if x >= 0 else 0

def train_perceptron(inputs, desired_outputs, learning_rate, epochs):
    global w1, w2, b
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            target_output = desired_outputs[i]
            output = activate(w1 * A + w2 * B + b)
            error = target_output - output
            w1 += learning_rate * error * A
            w2 += learning_rate * error * B
            b += learning_rate * error
            total_error += abs(error)
        if total_error == 0:
            break


inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
desired_outputs = [0, 1,1,0]
learning_rate = 0.1
epochs = 100

train_perceptron(inputs, desired_outputs, learning_rate, epochs)

for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(w1 * A + w2 * B + b)
    print(f"Input: ({A}, {B})  Output: {output}")
print(f"{w1} {w2}")