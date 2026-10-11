import random

import numpy as np


class Value:
    def __init__(self, value, children=(), op='', label='', grad=0):
        self.value = value
        self._prev = children
        self._backward = lambda: None
        self.grad = grad
        self._op = op
        self.label = label

    def __repr__(self):
        return f'{self.value}'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value - other.value, (self, other), '-')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        if other.value == 0:
            raise ZeroDivisionError('Division with zero is not allowed')

        out = Value(self.value / other.value, (self, other), '/')

        def _backward():
            self.grad += out.grad * (other.value**-1)
            other.grad += out.grad * (self.value * (-1 / other.value**2))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.value)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def dfs(node):
            if node in visited:
                return

            visited.add(node)

            for child in node._prev:
                dfs(child)

            topo.append(node)

        dfs(self)
        self.grad = 1

        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, n):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        aggr = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        activated = aggr.tanh()
        return activated

    def __repr__(self):
        return f'Weights: {[w for w in self.w]}, Bias: {self.b}'


class Layer:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out
        self.neurons = [Neuron(self.inp) for _ in range(self.out)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def __repr__(self):
        out_string = [f'Neuron : {neuron}' for neuron in self.neurons]
        return ',\n'.join(out_string)


class MLP:
    def __init__(self, ins, outs):
        sizes = [ins] + outs
        self.layers = [Layer(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __repr__(self):
        out_string = [f'Layer{i}:\n{layer}' for i, layer in enumerate(self.layers, start=1)]
        return ',\n\n'.join(out_string)

    def zero_grad(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.b.grad = 0
                for w in neuron.w:
                    w.grad = 0

    def update_weights(self, lr=0.01):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.b.value -= lr * neuron.b.grad
                for w in neuron.w:
                    w.value -= lr * w.grad


def MSE(actual, pred):
    if len(actual) != len(pred):
        raise ValueError('Shape mismatch')

    err_sq_sum = sum(((a - p) * (a - p) for a, p in zip(actual, pred)), Value(0))
    return err_sq_sum / len(actual)
