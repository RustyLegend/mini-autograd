# Mini Autograd

Mini Autograd is a small educational project that implements a minimal automatic differentiation engine from scratch in Python. It is inspired by the core ideas behind libraries such as PyTorch, but keeps the implementation compact and readable so the underlying mechanics of backpropagation are easier to understand.

The project builds a scalar-valued `Value` object, records operations into a computation graph, and performs reverse-mode automatic differentiation through that graph. On top of this engine, it includes simple neural network components such as neurons, layers, a multi-layer perceptron, and mean squared error loss.

## Project Goals

The main goal of this project is learning. It is not intended to be a production-ready machine learning framework. Instead, it focuses on making the following concepts clear:

- how values can track the operations that created them
- how a computation graph is built during the forward pass
- how gradients are propagated backward through the graph
- how simple neural network layers can be built from scalar operations
- how gradient descent updates model parameters

## Features

- Scalar automatic differentiation using a custom `Value` class
- Support for basic operations such as addition, subtraction, multiplication, and division
- Hyperbolic tangent activation function
- Reverse-mode backpropagation through a topologically sorted computation graph
- Simple `Neuron`, `Layer`, and `MLP` abstractions
- Mean squared error loss function
- Basic training loop support through gradient zeroing and weight updates

## Project Structure

```text
micrograd/
├── mini_autograd.py      # Reusable Python module with the autograd engine and MLP implementation
├── mini-autograd.ipynb   # Notebook version for learning and experimentation
├── examples/
│   └── xor_demo.py       # Runnable XOR training example
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

The notebook is useful for step-by-step experimentation, while `mini_autograd.py` makes the project easier to import, test, and run from the command line.

## Core Components

### `Value`

The `Value` class is the foundation of the project. Each `Value` stores:

- its numeric value
- its gradient
- the previous values that produced it
- the operation used to create it
- a backward function for propagating gradients

When operations are performed on `Value` objects, new `Value` objects are created and connected to their inputs. Calling `backward()` on a final output value walks through the graph in reverse and computes gradients for all dependent values.

### `Neuron`

A `Neuron` contains a list of weights and a bias. During the forward pass, it computes a weighted sum of the inputs and applies a `tanh` activation.

### `Layer`

A `Layer` is a collection of neurons. It takes an input vector and returns the output of each neuron in the layer.

### `MLP`

The `MLP` class stacks multiple layers to form a simple feedforward neural network. It also includes helper methods for clearing gradients and updating weights using gradient descent.

### `MSE`

The `MSE` function computes mean squared error between actual and predicted values. This is useful for training the MLP on small regression or toy classification-style problems.

## Example Usage

```python
from mini_autograd import MLP, MSE, Value

model = MLP(2, [4, 4, 1])

xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]

ys = [
    Value(0.0),
    Value(1.0),
    Value(1.0),
    Value(0.0),
]

for epoch in range(200):
    preds = [model(x)[0] for x in xs]
    loss = MSE(ys, preds)

    model.zero_grad()
    loss.backward()
    model.update_weights(lr=0.1)

    if epoch % 10 == 0:
        print(epoch, loss.value)
```

This example trains a small neural network on the XOR pattern. The model receives two inputs and learns to output a high value when the inputs are different and a low value when they are the same.

## Quick Start

Clone the project, install the dependencies, and run the demo:

```bash
pip install -r requirements.txt
python3 examples/xor_demo.py
```

The demo trains a small multi-layer perceptron on the XOR pattern and prints the loss during training followed by the final predictions.

## Notes and Limitations

This project intentionally keeps the implementation small. As a result, it has some limitations:

- It operates on scalar values rather than tensors.
- It does not support fully vectorized or batched computation.
- Inputs are typically passed one sample at a time.
- It is designed for small experiments and educational use.
- Gradients accumulate, so they should be cleared before each training step.

These constraints are acceptable for the purpose of understanding how automatic differentiation and backpropagation work at a low level.

## Requirements

The notebook uses Python and NumPy.

```bash
pip install -r requirements.txt
```

If you are using JupyterLab instead of the classic notebook interface, install it with:

```bash
pip install jupyterlab
```

## Running the Project

Start Jupyter Notebook or JupyterLab from the project directory:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open `mini-autograd.ipynb` and run the cells in order.

## Learning Context

This project is useful for understanding the building blocks of modern deep learning frameworks. Although real frameworks operate on tensors and include many performance optimizations, the same fundamental ideas are present here: build a graph during the forward pass, compute gradients during the backward pass, and update parameters using those gradients.
