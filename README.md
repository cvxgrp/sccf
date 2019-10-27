# sccf

`sccf` is a CVXPY extension for (approximately) minimizing a sum of clipped convex functions.
The algorithms used are described in [our paper](http://web.stanford.edu/~boyd/papers/min_sum_clipped_convex.html).

## Installation

Clone the repository, then run:
```bash
pip install .
```

## Usage
`sccf` exposes a function `minimum` and a class `Problem`.
One constructs an `objective` as a sum of terms that have the form `sccf.minimum(expr, constant)`,
then constructs a `Problem` object as `sccf.Problem(objective, constraints)`.
Then one calls the `solve` method, which has the following signature:
```python
def solve(self, method="alternating", *args, **kwargs):
    """Approximately solve the problem using alternating, convex-concave, or L-BFGS.

    Parameters
    ==========
    method : str
        alternating, cvx_ccv, or lbfgs
    args, kwargs: sent to underlying solve method and cvxpy.
    """
```

For example:
```python
import sccf
import cvxpy as cp

x = cp.Variable(10)
objective = sccf.minimum(cp.sum_squares(x), 1.0) + sccf.minimum(cp.sum_squares(x - .1), 1.0)
constraints = [cp.sum(x) == 1.0, x >= 0, x <= 1]
prob = sccf.Problem(objective, constraints)
prob.solve()
```

## Examples
For the examples in the paper, and more, see the `examples` folder.

## Run tests
Run:
```bash
python -m unittest
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Citing
If you use `sccf` in your research, please consider citing [our paper](http://web.stanford.edu/~boyd/papers/min_sum_clipped_convex.html):
```
@article{barratt2019minimizing,
    title={Minimizing a Sum of Clipped Convex Functions},
    author={Barratt, Shane and Angeris, Guillermo and Boyd, Stephen},
    journal={arXiv},
    year={2019}
}
```
