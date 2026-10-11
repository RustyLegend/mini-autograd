import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_autograd import MLP, MSE, Value


def main():
    random.seed(42)

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

    for epoch in range(300):
        preds = [model(x)[0] for x in xs]
        loss = MSE(ys, preds)

        model.zero_grad()
        loss.backward()
        model.update_weights(lr=0.1)

        if epoch % 50 == 0:
            print(f'epoch {epoch:03d} | loss {loss.value:.6f}')

    print('\nFinal predictions:')
    for x, y in zip(xs, ys):
        pred = model(x)[0]
        values = [v.value for v in x]
        print(f'{values} -> pred: {pred.value:.4f}, target: {y.value}')


if __name__ == '__main__':
    main()
