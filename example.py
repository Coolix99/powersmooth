import numpy as np
import matplotlib.pyplot as plt

from powersmooth.powersmooth import (
    powersmooth_general,
    upsample_with_mask,
    upsample_with_exact_data_inclusion,
)

if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    rng = np.random.default_rng(seed=42)
    perturbation = rng.uniform(-0.05, 0.05, size=x.size - 2)
    x[1:-1] += perturbation
    x = np.sort(x)
    y = np.sin(x) + 0.5 * np.random.randn(30)

    smooth_y_general = powersmooth_general(x,y, {1:0.0, 2:1e-3, 3:1e-3})

    x_up, y_up, mask_up = upsample_with_mask(x, y, dx=0.1)
    smooth_y_general_up = powersmooth_general(x_up, y_up, {1:0.0, 2:1e-3, 3:1e-3}, mask_up)

    x_dense, y_dense, mask, inserted_mask = upsample_with_exact_data_inclusion(x, y, dx=0.1)

    smooth = powersmooth_general(x_dense, y_dense, weights={1:0.0, 2:1e-3, 3:1e-3}, mask=mask)

    # remove the inserted points to return to original uniform grid
    x_final = x_dense[~inserted_mask]
    y_final = smooth[~inserted_mask]
    
    print(np.diff(x_final))
    print(x_final.shape)
    print(y_final.shape)

    plt.plot(x, y, label="Original")
    plt.plot(x, smooth_y_general, label="Smoothed (general)")
    plt.plot(x_up, smooth_y_general_up, label="Smoothed (upsampled)")
    plt.plot(x_final, y_final, label="Smoothed (uniform)")
    plt.legend()
    plt.show()
