"""
# Created by ashish1610dhiman at 06/04/23
Contact at ashish1610dhiman@gmail.com
"""

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from functools import partial
import torch
from torch import nn
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt



# assert issubclass(PyroModule[nn.Linear], nn.Linear)
# assert issubclass(PyroModule[nn.Linear], PyroModule)

import matplotlib.pyplot as plt

import argparse

import os
from functools import partial
import torch
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

from pyro.nn import PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(1)


def ashish_bayes_regr(X, y, NUM_ITER=3600):
    x_data = torch.tensor(X.values, dtype=torch.float)
    y_data = torch.tensor(y.values, dtype=torch.float)

    # set up regression model
    linear_reg_model = PyroModule[nn.Linear](X.shape[1], 1)

    # Define loss and optimize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.035)
    num_iterations = 3000 if not smoke_test else 2

    def train():
        # run the model forward on the data
        y_pred = linear_reg_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        return loss

    print("Fitting Bayes Regression")
    for j in range(NUM_ITER):
        loss = train()
        if (j + 1) % 300 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())

    return linear_reg_model,x_data,y_data



def ashish_bayes_regr_svi(X, y,n_samples=100, NUM_ITER=3600):
    x_data = torch.tensor(X.values, dtype=torch.float)
    y_data = torch.tensor(y.values, dtype=torch.float)

    class BayesianRegression(PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = PyroModule[nn.Linear](in_features, out_features)
            self.linear.weight = PyroSample(dist.Normal(0.0, 0.1).expand([out_features, in_features]\
                                                                         ).to_event(out_features))
            self.linear.bias = PyroSample(dist.Normal(-0.2, 0.5).expand([out_features]).to_event(1))

        def forward(self, x, y=None):
            sigma = pyro.sample("sigma", dist.Uniform(0.,2.0))
            mean = self.linear(x).squeeze(-1)
            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
            return mean

    # initialise guide
    regr_model = BayesianRegression(X.shape[1], 1)
    guide = AutoDiagonalNormal(regr_model)  # unobserved parameters in the model as a Gaussian with diagonal covariance

    # optimise ELBO
    adam = pyro.optim.Adam({"lr": 0.035})
    svi = SVI(regr_model, guide, adam, loss=Trace_ELBO())

    print("Optimising ELBO loss")
    pyro.clear_param_store()
    for j in range(NUM_ITER):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 300 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(X)))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    print ("Quantiles for params")
    print (guide.quantiles([0.25, 0.5, 0.75]))

    #sample
    predictive = Predictive(regr_model, guide=guide, num_samples=n_samples,
                            return_sites=("linear.weight","sigma","linear.bias", "obs", "_RETURN"))
    samples = predictive(x_data)

    param_samples = pd.DataFrame()
    for key, val_torch in samples.items():
        val_np = val_torch.reshape(val_torch.shape[0], val_torch.shape[-1]).detach().cpu().numpy()
        if "linear" in key or "sigma" in key:
            param_samples[[f"{key}_{i}" for i in range(val_np.shape[1])]] = val_np

    return samples,param_samples


