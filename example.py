import numpy as np
import matplotlib.pyplot as plt

from powersmooth.powersmooth import (
    powersmooth_general,
    upsample_with_mask,
    upsample_to_uniform,
)

if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    y = np.sin(x) + 0.5 * np.random.randn(30)

    smooth_y_general = powersmooth_general(x,y, {1:0.0,2:0.005})

    x_up, y_up, mask_up = upsample_with_mask(x, y, dx=0.1)
    smooth_y_general_up = powersmooth_general(x_up, y_up, {1:0.0, 2:1e-6, 3:1e-6}, mask_up)

    x_uniform, y_uniform, mask_uniform = upsample_to_uniform(x, y, dx=0.1)
    smooth_y_uniform = powersmooth_general(x_uniform, y_uniform, {1:0.0, 2:1e-6, 3:1e-6}, mask_uniform)

    plt.plot(x, y, label="Original")
    plt.plot(x, smooth_y_general, label="Smoothed (general)")
    plt.plot(x_up, smooth_y_general_up, label="Smoothed (upsampled)")
    plt.plot(x_uniform, smooth_y_uniform, label="Smoothed (uniform)")
    plt.legend()
    plt.show()
