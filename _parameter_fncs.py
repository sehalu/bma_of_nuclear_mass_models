# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:19:17 2020

@author: sehalu

Parameter functions/methods

"""

# Standard libraries
import numpy as np
from numpy.linalg import pinv
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# Own modules
from plotting_helpers import plot_parameter_Gaussian as plot_mvn
from plotting_helpers import plot_parameter_uniform as plot_uniform


plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 12})

width = 5.90666  # inches = textwidth in standard(?) LaTeX article

# =============================================================================

def zellner_g_prior(self):

    assert self.phi is not None, 'Cannot compute Zellner g without phi!'

    # If g not given, use number of measurements
    try:
        self.zellner_g
    except AttributeError:
        self.zellner_g = len(self.phi)
    
    if self.zellner_g in ['D', 'd', None]:
        self.zellner_g = len(self.phi)
    else:
        assert type(self.zellner_g) in [int, float], 'Zellner g not correct!'

    # Zellner is N(0, g [phi^T*phi]^-1)
    return (np.zeros(self.phi.shape[1]),
            self.zellner_g * pinv(self.phi.T @ self.phi, hermitian=True)
           )


# =============================================================================

def get_parameter_prior(self):
    '''
    Small function to get information about parameter prior distribution
    '''

    # If type is not set then the rest doesn't make sense
    try:
        sort = self.parameter_prior_type
    except AttributeError:
        print('No prior!')
        return None

    if sort in ['Gaussian', 'Gaussian_Zellner']:
        mu = sigma = None
        try:
            mu = self.parameter_prior_mu
        except AttributeError:
            print('Prior mean value "mu" is "None"!')
            pass
        try:
            sigma = self.parameter_prior_sigma
        except AttributeError:
            print('Prior covariance matrix "sigma" is "None"!')
            pass

        return (sort, mu, sigma)

    if sort == 'uniform':
        ranges = None
        try:
            ranges = self.parameter_prior_range
        except AttributeError:
            print('Prior ranges is "None"!')
            pass

        return (sort, ranges)

# =============================================================================

def get_parameter_posterior(self):
    '''
    Small function to get information about parameter posterior
    distribution.
    '''

    # If type is not set then the rest doesn't make sense
    try:
        sort = self.parameter_posterior_type
    except AttributeError:
        return None

    if sort == 'Gaussian':
        mu = sigma = None
        try:
            mu = self.parameter_posterior_mu
            sigma = self.parameter_posterior_sigma
        except AttributeError:
            pass

        return (sort, mu, sigma)

    if sort=='truncated_Gaussian':
        ranges = mu = sigma = None
        try:
            ranges = self.parameter_prior_range
            mu = self.parameter_posterior_mu
            sigma = self.parameter_posterior_sigma
        except AttributeError:
            pass

        return (sort, mu, sigma, ranges)

# =============================================================================

def plot_parameter_prior(self):

    prior = self.get_prior()

    if prior[0] in ['Gaussian', 'Gaussian_Zellner']:
        try:
            names = self.parameters
        except AttributeError:
            assert self.K is not None, 'Number of parameters is "None"!'
            names = [str(i+1) for i in range(self.K)]
        return plot_mvn(names, prior[1], prior[2])
    elif prior[0] in ['uniform']:
        try:
            names = self.parameters
        except AttributeError:
            assert self.K is not None, 'Number of parameters is "None"!'
            names = [str(i+1) for i in range(self.K)]
        return plot_uniform(names, prior[1])
    else:
        return -1

def plot_parameter_posterior(self):

    posterior = self.get_posterior()

    if posterior[0]=='Gaussian':
        try:
            names = self.parameters
        except AttributeError:
            assert self.K is not None, 'Number of parameters is "None"'
            names = [str(i+1) for i in range(self.K)]
        return plot_mvn(names, posterior[1], posterior[2])
    elif posterior[0]=='truncated_Gaussian':
        try:
            names = self.parameters
        except AttributeError:
            assert self.K is not None, 'Number of parameters is "None"'
            names = [str(i+1) for i in range(self.K)]
        return plot_mvn(names, posterior[1], posterior[2], posterior[3])
