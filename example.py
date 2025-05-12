import numpy as np
import matplotlib.pyplot as plt

from powersmooth.powersmooth import  powersmooth_general,upsample_with_mask

if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    y = np.sin(x) + 0.5 * np.random.randn(30)

    smooth_y_general = powersmooth_general(x,y, {1:0.0,2:0.005})

    x_up,y_up,mask_up=upsample_with_mask(x,y,dx=0.1)
    smooth_y_general_up=powersmooth_general(x_up,y_up, {1:0.0,2:1e-6,3:1e-6},mask_up)

    plt.plot(x, y, label="Original")
    plt.plot(x, smooth_y_general, label="Smoothed (general)")
    plt.plot(x_up, smooth_y_general_up, label="Smoothed (general upsampled)")
    plt.legend()
    plt.show()
