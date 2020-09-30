# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:33:10 2020

@author: sehalu

File for model functions

NumPy needed explicitly, pandas implicitly as stuff sent to it are thereof,
such as design matrix. Well, it ought to be.

"""

# Standard libraries
import numpy as np
import pandas as pd

# Plotting helpers
# from plotting_helpers import plot_model_probability

# =============================================================================
# Get stuff
# =============================================================================

def get_subcorr(corr_mat: np.ndarray, model_nbr: int) -> np.ndarray:

    assert corr_mat.shape[0] == corr_mat.shape[1]

    model_dim = corr_mat.shape[0]

    # Boolean array to acces basis functions by turning the model number into
    # binary form, e.g. k=18 -> 10010 -> phi0 and phi3
    # Note the flip, it shouldn't matter for calculations but for analysis.
    k_bool = np.array([int(i) for i in f'{model_nbr:0{model_dim}b}'],
                      dtype=bool)

    # Correlation (sub)matrix
    corr_mat_k = corr_mat[k_bool, :][:, k_bool]

    assert corr_mat_k.shape[1] == corr_mat_k.shape[0] == sum(k_bool)

    return corr_mat_k


def get_the_model(k: int, K: int, parameters: list, phi: pd.DataFrame,
                  with_prior: bool=False, prior_type: str=None, prior: tuple=None):
    # Boolean array to acces basis functions by turning the model number into
    # binary form, e.g. k=18 -> 10010 -> phi0 and phi3
    # Note the flip, it shouldn't matter for calculations but for analysis.
    k_bool = np.array([int(i) for i in f'{k:0{K}b}'], dtype=bool)

    # Model_k is the  corresponding basis functions
    model_k = parameters[k_bool]
    phi_k = phi[model_k]

    if with_prior:
        # If uniform/flat prior, return also prior[min,max][model]
        if prior_type=='uniform':
            assert prior.ndim==2
            Delta_alpha_k = prior[k_bool]
            return phi_k, Delta_alpha_k
        # If Gaussian/normal prior, return also p(alpha_k) ~ N(mu_k, Sigma_k)
        elif prior_type in ['Gaussian', 'Zellner']:
            mu_k = prior[0][k_bool]
            sigma_k = prior[1][k_bool, :][:, k_bool]
            return phi_k, (mu_k, sigma_k)
    else:
        return phi_k

    return -1


def get_model(self, k: int=None, K: int=None, parameters: list=None,
              phi: pd.DataFrame=None, with_prior: bool=False,
              prior_type: str=None, prior_values: tuple=None):
    '''
    Get design matrix corresponding to model k, and priors if desired.
    '''

    # Make sure everything is in order
    if k is None:
        assert self.nbr_of_models is not None, 'No number of models!'
        k = self.nbr_of_models

    if K is None:
        assert self.K is not None, 'No model dimension!'
        K = self.K

    if parameters is None:
        assert self.parameters is not None, 'No parameters!'
        parameters = self.parameters

    if phi is None:
        assert self.phi is not None, 'No design matrix!'
        phi = self.phi

    if with_prior:
        if prior_type is None:
            assert self.parameter_prior_type is not None, 'No parameter prior type!'
            prior_type = self.parameter_prior_type

        if prior_type == 'uniform':
            if prior_values is None:
                assert self.parameter_prior_range is not None, 'No parameter prior range!'
                prior_values = self.parameter_prior_range
            else:
                assert len(prior_values)==3, 'Wrong shape of prior values!'
                prior_values = prior_values[0]
            assert prior_values.shape==(K, 2), 'Wrong shape of parameter prior ranges!'
        elif prior_type in ['Gaussian', 'Zellner']:
            if prior_values is None:
                assert self.parameter_prior_mu is not None, 'No parameter prior mu!'
                assert self.parameter_prior_sigma is not None, 'No parameter prior sigma!'
                prior_values = (self.parameter_prior_mu, self.parameter_prior_sigma)
            else:
                assert len(prior_values)==3, 'Wrong shape of prior values!'
                prior_values = (prior_values[1], prior_values[2])
            assert prior_values[0].shape==(K,), 'Wrong shape of parameter prior mu!'
            assert prior_values[1].shape==(K, K), 'Wrong shape of parameter prior sigma!'
    
    return get_the_model(k, K, parameters, phi, with_prior, prior_type, prior_values)


# =============================================================================
# Model priors
# =============================================================================

def correlation_matrix(phi: np.ndarray, K: int):
    '''
    Calculate the, Pearson, correlation matrix of the design matrix
    phi: design matrix
    K: number of basis functions
    '''
    assert phi is not None, 'No design matrix!'

    phi_cov = np.linalg.pinv(phi.T @ phi, hermitian=True)

    # Fix diagonal, can't have zeroes
    for k in range(K):
        phi_cov[k, k] = phi_cov[k, k] if phi_cov[k, k] > 1e-9 else 1e-9

    assert not np.isnan(phi_cov).any(), 'NaN in correlation covariance matrix!'

    # Parameter correlation matrix, Pearson
    phi_corr = np.empty(shape=(K, K), dtype=float)
    for k in range(K):
        for l in range(K):
            try:
                phi_corr[k, l] = phi_cov[k, l] /\
                    np.sqrt(phi_cov[k, k] * phi_cov[l, l])
            except FloatingPointError:
                print(phi_cov[k, l], phi_cov[k, k], phi_cov[l, l])

    assert not np.isnan(phi_corr).any(), 'NaN in correlation matrix!'

    return phi_corr


# =============================================================================
# Model class functions
# =============================================================================

def model_prior_uniform(self):
    '''
    Calculates the logarithm of a uniform model prior.
    p(M_k) = 1 / (2^K-1)
    where K is number of parameters, thus 2^K-1 = number of models
    '''
    assert self.nbr_of_models is not None, 'No number of models!'
    prior_log_pdf = -np.log(self.nbr_of_models) \
                    * np.ones(shape=(self.nbr_of_models,), dtype=float)

    prior_log_pdf = [ [i+1, plp] for i,plp in enumerate(prior_log_pdf) ]
    prior_log_pdf = pd.DataFrame({'model_number': [plp[0] for plp in prior_log_pdf],
                                  'log_pdf': [plp[1] for plp in prior_log_pdf]})
    prior_log_pdf['model_number'] = prior_log_pdf.astype(int)
    prior_log_pdf['log_pdf'] = pd.to_numeric(prior_log_pdf['log_pdf'])
    return prior_log_pdf


def model_prior_dilution(self):
    '''
    Calculates the logarithm of a dilution prior, using dilution factor p.
    p(M_k) = |corr(phi_k)|^p / (2^K-1)
    where phi is design matrix and K is number of parameters.
    If parameters of model k are uncorrelated, |corr(phi_k)|=1
    '''

    assert self.nbr_of_models is not None, 'No number of models exist!'
    assert self.K is not None, 'No number of basis functions exist!'
    assert self.phi is not None, 'No design matrix exists!'

    corr_mat = correlation_matrix(self.phi, self.K)

    prior_log_pdf = np.ones(shape=(self.nbr_of_models,), dtype=float)

    # Logarithm of (absolute value of) determinant of parameter correlation
    # matrix for each model
    for i in range(self.nbr_of_models):
        corr_mat_k = get_subcorr(corr_mat, i+1)
        _, log_det_corr_k = np.linalg.slogdet(corr_mat_k)
        prior_log_pdf[i] = log_det_corr_k

    # Divide by number of models, log -> subtract
    prior_log_pdf -= np.log(self.nbr_of_models)

    # Raise correlation determinant to dilution factor
    p = self.dilution_factor if self.dilution_factor is not None else 1.
    prior_log_pdf *= p

    prior_log_pdf = [ [i+1, plp] for i,plp in enumerate(prior_log_pdf) ]
    prior_log_pdf = pd.DataFrame({'model_number': [plp[0] for plp in prior_log_pdf],
                                  'log_pdf': [plp[1] for plp in prior_log_pdf]})

    prior_log_pdf['log_pdf'] = pd.to_numeric(prior_log_pdf['log_pdf'])

    return prior_log_pdf

# =============================================================================
# Plot
# =============================================================================

# def plot_model_prior(self):
#     '''
#     Plot model prior probability distribution
#     '''

#     assert self.model_prior is not None, 'No model prior!'

#     fig, ax = plot_model_probability(self.model_prior)
