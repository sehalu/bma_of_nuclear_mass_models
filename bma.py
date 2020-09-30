'''
Created on 17 Aug 2020
Author: sehalu

Based on / copied from previous work
'''

import numpy as np
import pandas as pd
import sys

# Minimum logarithm, below this: underflow
# Un-normalised logarithm of minimum value
# Values smaller than this are effectively zero, precision can't be ensured.
LOG_EPS = np.log(sys.float_info.min) + np.log(sys.float_info.epsilon)

np.seterr(under='warn')


# ============================================================================
# ============================================================================

def evidences_uniform(D: pd.DataFrame, E: pd.DataFrame, phi: pd.DataFrame,
                      prior: np.ndarray):
    '''
    Compute evidence of models, marginal model likelihood, given data and model.
    In the case of uniform parameter prior, i.e.
        prob(alpha) = Heaviside(Delta_alpha)/vol(Delta_alpha)

    Input/Parameters:
    -------------------
    Dataframe: pandas.dataframe
        Includes D: data vector and Phi: design matrix
    E: array_like
        Covariance matrix of measurements
    Prior: array_like
        Parameter prior, a set of [min, max] for each parameter

    Returns:
    -------------------
    evidences: array_like
        Logarithm of model evidences
    ppp: array_like
        Posterior parameter probability given by a vector of optimal values in
        least-squares sense and a covariance matrix
    '''

    # Data and model dimensions
    N = len(D)
    K = phi.shape[1]

    if E.ndim == 1:
        E = np.diag(E)
    E_inv = np.linalg.pinv(E, hermitian=True)
    _, E_logdet = np.linalg.slogdet(E)

    # logarithm of volume of parameter prior hypercube
    logVol_alpha = sum(np.log(prior[1]-prior[0]))

    Psi = np.dot(phi.T, np.dot(E_inv, phi))
    Psi_inv = np.linalg.pinv(Psi, hermitian=True)
    _, Psi_logdet = np.linalg.slogdet(Psi)

    # least-squares optimum solution for parameters, Psi alpha = y
    y = np.dot(phi.T, np.dot(E_inv, D))
    alpha_hat = np.dot(Psi_inv, y)

    # Check if parameter prior excludes the least-squares optimal solution
    # Then the posterior, or rather Occam factor, must be identically zero.
    # We have removed the possibility of the optimum value by our ignorance.
    if not (np.all(prior[0] < alpha_hat) and
            np.all(alpha_hat < prior[1])):
        logOccam = -np.inf  # logarithm, ln(0)=-inf <=> exp(-inf)=0
        print(f'\n\tBad priors: Excludes least-squares solution.')
    else:
        logOccam = .5*K*np.log(2*np.pi) - .5*Psi_logdet - logVol_alpha

    # Least-squares solution of quadratic form
    x = D - np.dot(phi, alpha_hat)
    chi2_min = np.dot(x.T, np.dot(E_inv, x))

    # Maximum Likelihood Estimator
    logL_max = -.5*chi2_min - .5*N*np.log(2*np.pi) - .5*E_logdet

    # Logarithm of model evidence
    logEv = logL_max + logOccam

    # parameter posterior
    parameter_posterior = {'mu': alpha_hat, 'sigma': Psi_inv}

    return logEv, parameter_posterior


def evidences_Gaussian(D: pd.DataFrame, E: pd.DataFrame, phi: pd.DataFrame,
                       mu: np.ndarray, sigma: np.ndarray):
    '''
    Computes evidence of models, marginal model likelihood, given data and model.
    In the case of normal parameter prior, i.e.
        prob(alpha) = N(mu, Sigma)

    Input/Parameters:
    --------------------
    Dataframe: pandas.dataframe
        Includes D: data vector and Phi: design matrix
    E: array_like
        Covariance matrix of measurements
    Prior: dict
        Parameter prior, an expected vector and covariance matrix [mu, Sigma]

    Returns:
    ------------------
    evidences: array_like
        Logarithm of model evidences
    ppp: dict
        Posterior parameter probability given by N(post_mu, post_sigma)
    '''

    N = len(D)

    # Check if data covariance is vector
    if E.ndim == 1:
        E = np.diag(E)
    E_inv = np.linalg.pinv(E, hermitian=True)

    sigma_inv = np.linalg.pinv(sigma, hermitian=True)

    # Marginal model likelihood, model evidence, covariance matrix
    Epost = E + np.dot(phi, np.dot(sigma, phi.T))
    Epost_inv = np.linalg.pinv(Epost, hermitian=True)
    _, Epost_logdet = np.linalg.slogdet(Epost)

    # Logarithm of model evidence
    x = D - np.dot(phi, mu)
    logEv = -.5*np.dot(x.T, np.dot(Epost_inv, x)) \
        - .5*N*np.log(2*np.pi) - .5*Epost_logdet

    # Posterior parameter covariance matrix, Lambda
    Lambda = np.linalg.pinv(sigma_inv + np.dot(phi.T, np.dot(E_inv, phi)),
                            hermitian=True)

    # parameter posterior
    mupost = np.dot(Lambda, np.dot(phi.T, np.dot(E_inv, D)) +
                    np.dot(sigma_inv, mu))
    parameter_posterior = {'mu': mupost, 'sigma': Lambda}

    return logEv, parameter_posterior


def evidences(D: pd.DataFrame, E: np.ndarray, phi: pd.DataFrame,
              parameter_prior_type: str, parameter_prior: tuple):
    '''
    Simple choice method, send everything here and choose uniform or Gaussian

    Input/Parameters:
    -------------------
    DF: pandas.dataframe
        Includes D: data vector and Phi: design matrix
    E: array_like
        Data covariance matrix
    parameter_prior: dict
        type: str -- 'Uniform'/'flat' OR 'Gaussian'/'normal/
        range OR values: np.array OR dict

    Returns:
    -------------------
    evidences: array_like
        Logarithm of model evidences
    ppp: dict
        Posterior parameter probability given by a vector of optimal values in
        least-squares sense and a covariance matrix
        OR
        Posterior parameter probability given by N(post_mu, post_sigma)
    '''

    if parameter_prior_type == 'uniform':
        return evidences_uniform(D, E, phi, parameter_prior)
    elif parameter_prior_type in ['Gaussian', 'Zellner']:
        return evidences_Gaussian(D, E, phi, parameter_prior[0], parameter_prior[1])

# =============================================================================
# =============================================================================

def lse(log_x: np.ndarray) -> float:
    '''
    Own log_sum_exp function, encounter underflow etc.
    This will (hopefully) remedy this as we need alot of
    precision for unlikely models(?).
    LSE is a soft-max function, i.e.
        max(x) <= LSE(x) <= max(x) + ln(n)
    where n = len(x)
    '''
    lnx = np.atleast_1d(log_x[:]).astype(float)
    max_lnx = max(log_x)

    lnx = [l-max_lnx for l in lnx if l-max_lnx>LOG_EPS]

    lse_x = np.log(sum(np.exp(lnx)))
    lse_x += max_lnx

    if lse_x<LOG_EPS:
        print('log_sum_exp returns too small elements.')

    return lse_x


# =============================================================================

def posterior_model_probability(log_evidences: np.array, model_prior: pd.DataFrame,) -> np.array:
    '''
    Compute the logarithm of posterior model probability from logarithm of
    model evidences.

    Parameters/Input:
    --------------
    log_evidences: np.array
        Logarithm of model evidences, array/vector/list
    mp_type: str
        model prior probability type, 'uniform' or 'dilution'

    Returns:
    -------------
    log_pmp: np.array
        Logarithm of posterior model probability
    '''

    log_prior = model_prior['log_pdf']

    # Assert same length
    assert log_evidences.ndim == log_prior.ndim == 1
    assert log_evidences.shape == log_prior.shape

    log_pmp = log_evidences + log_prior
    log_pmp -= lse(log_pmp)

    # Rectify if there are inf or below precision
    log_pmp[log_pmp<=LOG_EPS] = -np.infty

    log_pmp_df = pd.DataFrame({'model_number':
                               [k+1 for k,_ in enumerate(log_pmp)],
                               'log_pdf': log_pmp})

    return log_pmp_df

# =============================================================================
# =============================================================================


# =============================================================================
# =============================================================================
