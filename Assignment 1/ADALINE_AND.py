import numpy as np


class Adaline_And:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return np.sign(x)

    def predict(self, x):
        net_input = np.dot(x, self.weights) + self.bias
        return self.activation(net_input)

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error


def main():
    # Training data for AND gate (bipolar inputs and target outputs)
    X_and = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    input_size = X_and.shape[1]
    learning_rate = 0.1
    epochs = 100

    # Create and train ADALINE models for AND gates
    and_model = Adaline_And(input_size, learning_rate, epochs)
    and_model.train(X_and, y_and)

    # Test the trained models
    test_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    print("AND Gate Predictions:")
    for i in range(len(test_inputs)):
        prediction = and_model.predict(test_inputs[i])
        print(f"Input: {test_inputs[i]}, Prediction: {prediction}")


if __name__ == "__main__":
    main()
