"""
The goal of this script is to see how to share parameters between models in lmfit
even when using prefixes
"""

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('MacOSX')
import lmfit


def my_model(x, a, b, c):
    # gaussian
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


true_params = {'model_1':
                   {'a': 2.1,
                    'b': 0.5,
                    'c': 0.2},
               'model_2':
                   {'a': 2.1,
                    'b': 4.5,
                    'c': 0.8}, }

x = np.linspace(-2, 10, 1000)
synthetic_data = (my_model(x, true_params['model_1']['a'],
                           true_params['model_1']['b'],
                           true_params['model_1']['c']) +
                  my_model(x, true_params['model_2']['a'],
                           true_params['model_2']['b'],
                           true_params['model_2']['c']) +
                  0 * np.random.normal(0, 0.1, len(x)))

# plt.plot(x, synthetic_data, 'k+')

# %%
model_1 = lmfit.Model(my_model, prefix='m1_')
model_2 = lmfit.Model(my_model, prefix='m2_')

model = model_1 + model_2
params = model.make_params()

# Add global_c to the params
params.add('global_c', value=0.2, vary=True, min=0, max=1)

# Set the expressions for the parameters
params['m1_c'].set(expr='global_c')
params['m2_c'].set(expr='global_c')

# fix all the other parameters
params['m1_a'].set(value=true_params['model_1']['a'], vary=False)
params['m1_b'].set(value=true_params['model_1']['b'], vary=False)
params['m2_a'].set(value=true_params['model_2']['a'], vary=False)
params['m2_b'].set(value=true_params['model_2']['b'], vary=False)

params.pretty_print()

result = model.fit(synthetic_data, params, x=x)
result.plot()

result.summary()
