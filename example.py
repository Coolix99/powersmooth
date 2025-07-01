import numpy as np
import matplotlib.pyplot as plt

from powersmooth.powersmooth import (
    powersmooth_general,
    powersmooth_upsample,
    powersmooth_on_uniform_grid,
)

if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    rng = np.random.default_rng(seed=42)
    perturbation = rng.uniform(-0.05, 0.05, size=x.size - 2)
    x[1:-1] += perturbation
    x = np.sort(x)
    y = np.sin(x) + 0.5 * rng.standard_normal(x.shape)

    weights = {1: 0.0, 2: 1e-3, 3: 1e-3}

    # Original non-uniform smoothing
    smooth_y_general = powersmooth_general(x, y, weights)

    # Smoothed on upsampled grid
    x_up, y_up_smooth,_ = powersmooth_upsample(x, y, weights, dx=0.1)

    # Smoothed on uniform grid, stripped of inserted points
    x_uniform, y_uniform_smooth = powersmooth_on_uniform_grid(x, y, weights, dx=0.1)

    # Plot all
    import matplotlib.pyplot as plt
    plt.plot(x, y, label="Original")
    plt.plot(x, smooth_y_general, label="Smoothed (general)")
    plt.plot(x_up, y_up_smooth, label="Smoothed (upsampled)")
    plt.plot(x_uniform, y_uniform_smooth, label="Smoothed (uniform grid)")
    plt.legend()
    plt.title("Comparison of Smoothing Methods")
    plt.show()

