import numpy as np
import pandas as pd
import matplotlib as mpl


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # number of x/input nodes defined
        self.hidden_size = hidden_size # number of h/hidden nodes defined
        self.output_size = output_size # number of y/output nodes defined

        # Initialize weights and biases
        self.weights_xh = np.random.rand(self.input_size, self.hidden_size) # assign random weights from input to hidden layers
        self.bias_hidden = np.zeros((1, self.hidden_size)) # bias for hidden layer  
        self.weights_hy = np.random.rand(self.hidden_size, self.output_size) # assign random weights from hidden to output layers
        self.bias_output = np.zeros((1, self.output_size)) # output bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        # Forward propagation
        self.input_data = input_data
        self.hidden_layer_input = np.dot(input_data, self.weights_xh) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hy) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)
        return self.output

    def backward(self, target, learning_rate):
        # Backpropagation
        error = target - self.output

        # Output layer
        output_delta = error * self.sigmoid_derivative(self.output)
        hidden_layer_output_transpose = self.hidden_layer_output.T
        self.weights_hidden_output += np.dot(hidden_layer_output_transpose, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # Hidden layer
        hidden_layer_error = np.dot(output_delta, self.weights_hy.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        input_data_transpose = self.input_data.T
        self.weights_xh += np.dot(input_data_transpose, hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    def train(self, input_data, target, learning_rate, epochs):
        for _ in range(epochs):
            self.forward(input_data)
            self.backward(target, learning_rate)

    def predict(self, input_data):
        return self.forward(input_data)

# Example usage
if __name__ == "__main__":
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Pandas
    target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])  # Updated target values for 2 output units

    input_size = 2 # input nodes
    hidden_size = 4 # hidden nodes
    output_size = 2  # Output nodes
    learning_rate = 0.7 # percent of data used as learning data 
    epochs = 10000

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        nn.train(input_data, target, learning_rate, 1)
        if (epoch + 1) % 1000 == 0:
            error = np.mean(np.square(target - nn.predict(input_data)))
            print(f"Epoch {epoch + 1}/{epochs}, Error: {error}")

    predictions = nn.predict(input_data)
    print("Final Predictions:")
    print(predictions)
