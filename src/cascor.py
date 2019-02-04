import numpy as np
from loss_functions import *


class HiddenUnit:
    def __init__(self, index, num_inputs, num_outputs, activation):
        self.index = index
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation

        self.inputs = np.zeros((self.num_inputs, 1))
        self.z = 0
        self.output = 0

        # are only calculated in the correlation maximization step
        # will be frozen after this hidden unit is added to the network
        self.input_weights = np.random.normal(
            0,
            np.sqrt(2.0 / self.num_inputs),
            (self.num_inputs, 1)
        )

        # gradients of the correlation w.r.t. input weights (dS/dw)
        self.grad_weights = np.zeros(self.input_weights.shape)

    def forward(self, inputs):
        assert (inputs.shape == self.input_weights.shape)

        # compute output = f(z); z = W_in * x
        self.inputs = inputs
        self.z = np.array(np.sum(self.input_weights * self.inputs))
        self.output = self.activation(self.z)

        return self.output

    def update_input_weights(self, update_rate):
        self.input_weights += update_rate * self.grad_weights

    def compute_correlation_derivative(self, sigma, d_act, hu_inputs, errors, errors_avg):
        self.grad_weights.fill(0)

        for wi in range(self.num_inputs):
            tmp = d_act * hu_inputs[:, wi]

            for o in range(self.num_outputs):
                self.grad_weights[wi] += sigma[o] * np.sum(tmp * (errors[:, o] - errors_avg[o]))

    def get_input_weights(self):
        return self.input_weights

    def to_string(self):
        return "[HU (%s -> %s)]" % (self.num_inputs, self.num_outputs)


class CascadeCorrelationNet:
    def __init__(self, num_inputs, num_outputs, activation):
        self.num_inputs = num_inputs + 1  # account for the bias unit
        self.num_outputs = num_outputs
        self.activation = activation

        self.hidden_units = []

        self.inputs = np.zeros((self.num_inputs, 1))
        self.outputs = np.zeros((self.num_outputs, 1))

        # output weights
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / (self.num_inputs + self.num_outputs)),
            (self.num_outputs, self.num_inputs)
        )

        # z = Wx
        self.z = np.zeros((self.num_outputs, 1))

        # dL/dw
        self.grad_weights = np.zeros(self.weights.shape)

    def forward_hidden_units(self, inputs):
        hu_outputs = np.zeros((self.num_hidden_units(), 1))

        for i, hu in enumerate(self.hidden_units):
            hu_input = np.concatenate((hu_outputs[:i], inputs, [[1]]), axis=0)
            hu_outputs[i] = hu.forward(hu_input)

        return hu_outputs

    def forward_candidate_hidden_unit(self, inputs, candidate_hidden_unit):
        # forward through previous hidden units
        prev_hu_outputs = self.forward_hidden_units(inputs)

        # forward through current candidate
        candidate_input = np.concatenate((prev_hu_outputs, inputs, [[1]]), axis=0)
        candidate_output = candidate_hidden_unit.forward(candidate_input)

        return candidate_output

    def forward(self, inputs):
        assert (inputs.shape == (self.num_inputs - 1, 1))  # account for the bias unit

        # compute output of each hidden unit
        hu_outputs = self.forward_hidden_units(inputs)

        # concatenate the entire input into a single column
        # hidden units outputs ++ network inputs ++ bias value
        self.inputs = np.concatenate((hu_outputs, inputs, [[1]]), axis=0)

        assert (self.inputs.shape[0] == self.weights.shape[1])

        self.z = np.dot(self.weights, self.inputs)
        self.outputs = self.activation(self.z)

        return self.outputs

    def backward(self, upstream):
        assert (upstream.shape == (self.num_outputs, 1))

        # upstream = dL/dy
        # dL/dw = dL/dy * dy/dz * dz/dw = upstream * d_activation(z) * inputs

        # gradients w.r.t. the weights
        self.grad_weights = (upstream * self.activation(self.z, derivate=True)) * self.inputs.T

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights

    def insert_hidden_unit(self, hidden_unit):
        # trainable output weights for current hidden unit
        hidden_unit_weights = np.random.normal(
            0,
            np.sqrt(2.0 / self.num_outputs),
            (self.num_outputs, 1)
        )

        # extend weights with current hidden unit's weights
        self.weights = np.concatenate((self.weights, hidden_unit_weights), axis=1)
        # extend gradients for current hidden unit's weights
        self.grad_weights = np.zeros(self.weights.shape)

        self.hidden_units.append(hidden_unit)

    def compute_correlation(self, inputs, labels, hidden_unit: HiddenUnit):
        correlation = np.zeros((self.num_outputs, 1))
        errors = np.zeros((len(inputs), self.num_outputs, 1))

        hu_input_size = self.num_hidden_units() + len(inputs[0]) + 1
        hu_outputs = np.zeros((len(inputs), 1))
        hu_z = np.zeros((len(inputs), 1))
        hu_inputs = np.zeros((len(inputs), hu_input_size, 1))

        net_outputs = np.zeros((len(inputs), self.num_outputs, 1))
        act = hidden_unit.activation

        for i, x in enumerate(inputs):
            # hidden unit's output for input x
            hu_outputs[i] = self.forward_candidate_hidden_unit(x, hidden_unit)
            # cache current inputs and z = Win * x
            hu_inputs[i] = hidden_unit.inputs
            hu_z[i] = hidden_unit.z

            # network's output for input x
            net_outputs[i] = self.forward(x)

            # target for input x
            target = np.zeros((10, 1))
            target[labels[i]] = 1

            # network's error for input x
            errors[i] = loss_func(net_outputs[i], target)

        # average activation of hidden unit, over all inputs
        hu_avg = np.average(hu_outputs)

        # average errors of each output, over all inputs
        errors_avg = np.average(errors, axis=0)

        # compute correlation of each output
        for o in range(self.num_outputs):
            correlation[o] = np.sum((hu_outputs - hu_avg) * (errors[:, o] - errors_avg[o]))

        sigma = np.sign(correlation)
        d_act = act(hu_z, derivate=True)

        return correlation, sigma, d_act, hu_inputs, errors, errors_avg

    def maximize_correlation(self, inputs, labels, hidden_unit: HiddenUnit):
        max_steps = 30
        i = 0
        prev_s = 0

        corr, sigma, d_act, hu_inputs, errors, errors_avg = self.compute_correlation(inputs, labels, hidden_unit)
        s = np.sum(np.abs(corr))

        while np.abs(s - prev_s) >= 0.005 or i < max_steps:
            print("maximize correlation [{}]: {}".format(i, s))
            i += 1

            hidden_unit.compute_correlation_derivative(sigma, d_act, hu_inputs, errors, errors_avg)

            hidden_unit.update_input_weights(0.1)

            prev_s = s

            corr, sigma, d_act, hu_inputs, errors, errors_avg = self.compute_correlation(inputs, labels, hidden_unit)
            s = np.sum(np.abs(corr))

        print("[DONE] maximize correlation [{}]: {} (err = {})".format(i, s, np.abs(s - prev_s)))

    def num_hidden_units(self):
        return len(self.hidden_units)

    def to_string(self):
        num_hu = self.num_hidden_units()

        s = "[CC (%s -> %s) | %s]\n" % (self.num_inputs, self.num_outputs, num_hu)

        for i in range(num_hu):
            s += "\t" + str(i + 1) + " : " + self.hidden_units[i].to_string() + "\n"

        return s
