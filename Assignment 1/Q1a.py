import numpy as np


class Adaline:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for the bias weight
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self, inputs, targets):
        for _ in range(self.epochs):
            for input_data, target in zip(inputs, targets):
                input_with_bias = np.insert(input_data, 0, 1)  # Insert bias input
                predicted = self.activate(np.dot(input_with_bias, self.weights))
                error = target - predicted
                self.weights += self.learning_rate * error * input_with_bias

    def predict(self, inputs):
        predictions = []
        for input_data in inputs:
            input_with_bias = np.insert(input_data, 0, 1)  # Insert bias input
            predicted = self.activate(np.dot(input_with_bias, self.weights))
            predictions.append(predicted)
        return predictions


# Training data for AND gate (bipolar inputs and target outputs)
and_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
and_targets = np.array([-1, -1, -1, 1])

# Training data for OR gate (bipolar inputs and target outputs)
or_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
or_targets = np.array([-1, 1, 1, 1])

# Create ADALINE models for AND and OR gates
and_adaline = Adaline(input_size=2)
or_adaline = Adaline(input_size=2)

# Train ADALINE models
and_adaline.train(and_inputs, and_targets)
or_adaline.train(or_inputs, or_targets)

# Test the trained models
test_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
and_predictions = and_adaline.predict(test_inputs)
or_predictions = or_adaline.predict(test_inputs)

print("AND Gate Predictions:", and_predictions)
print("OR Gate Predictions:", or_predictions)
