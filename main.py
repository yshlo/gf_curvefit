# -*- coding: utf-8 -*-
# 2015.2.12

import numpy as np
from scipy.optimize import curve_fit

class fitting():
    def __init__(self):
        self.param_init = [100., 100., 100.]
        self.method = "lm"

    def gauss(self, x, *p):
        mu, sigma, A = p
        return np.exp(-(x-mu)**2/(2.*sigma**2))*A

    def fitting_normal(self, datax, datay):
        coeff, var_matrix = curve_fit(self.gauss, datax, datay, p0=self.param_init, method=self.method)
        mu_opt =coeff[0]
        sigma_opt = coeff[1]
        Z_opt = coeff[2]
        return mu_opt, sigma_opt, Z_opt

if __name__ == '__main__':
    example_x = [-370, -290, -200, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    example_y = [0, 0, 0, 8, 9, 9,  8, 6, 5, 3, 1, 1, 0, 0]

    fit = fitting()
    coeffs = fit.fitting_normal(example_x, example_y)
    print("Optimal parameters: (mu_opt, sigma_opt, Z_opt) = ", coeffs)

    import matplotlib.pyplot as plt
    plt.plot(example_x, example_y)
    X = np.arange(-500, 1000)
    plt.plot(X, fit.gauss(X, coeffs[0], coeffs[1], coeffs[2]))
    plt.show()
