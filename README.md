# powersmooth

Python implementation of powersmooth (https://de.mathworks.com/matlabcentral/fileexchange/48799-powersmooth), invented by B. M. Friedrich.

This package provides derivative-based regularization smoothing for one-dimensional signals on **uniform and non-uniform grids**.


---

## Mathematical Formulation

Given data points $ (x_i, y_i) $, the smoothed signal $ u $ is computed as the minimizer of

$$
\min_u 
\; \| M (u - y) \|_2^2
\;+\;
\sum_{k} w_k \| D_k u \|_2^2
$$

where:

- $ M $ is a diagonal mask matrix controlling data fidelity
- $ D_k $ is a finite-difference approximation of the $ k $-th derivative
- $ w_k \ge 0 $ are regularization weights

This allows explicit penalization of first, second, or third derivatives.

The method works on:

- Non-uniform grids
- Uniform grids
- Upsampled grids with masked original data points

---

## Installation

### From PyPI

```bash
pip install powersmooth
```


### From GitHub

```bash
git clone https://github.com/Coolix99/powersmooth.git
cd powersmooth
pip install .
```

For development (editable install):

```bash
pip install -e .
```



---

## Quick Example

```python
import numpy as np
from powersmooth import powersmooth_general

# Generate noisy data
x = np.linspace(0, 10, 50)
y = np.sin(x) + 0.3 * np.random.randn(len(x))

# Penalize second derivative (curvature)
weights = {2: 1e-2}

y_smooth = powersmooth_general(x, y, weights)
```

---


## Requirements

- Python ≥ 3.9  
- NumPy  
- SciPy  

---

## License

MIT License
