
import numpy as np
import matplotlib.pyplot as plt

from powersmooth.powersmooth import powersmooth, powersmooth_upsampled

if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    y = np.sin(x) + 0.5 * np.random.randn(30)
    order = 1
    weight = 20.5
    factor = 4

    smooth_y_uniform = powersmooth(y, order, weight)
    upsampled_y_uniform = powersmooth_upsampled(y, order, weight * 200, factor)

    plt.plot(x, y, label="Original")
    plt.plot(x, smooth_y_uniform, label="Smoothed (uniform)")
    plt.plot(np.linspace(0, 10, len(upsampled_y_uniform)), upsampled_y_uniform, label="Upsampled (uniform)")
    plt.legend()
    plt.show()
