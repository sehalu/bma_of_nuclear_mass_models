'''
Script file for running a Bayesian analysis implemented by
    Sebastian Lundquist | id: sehalu | CID: seblund

Herein variable names will be specified and what priors to be used.
Then calling an analysis method within the model class. This will use
a Bayesian approach to statistics to calculate posterior distributions for
parameters and models.

Only linear models are implemented by author; to analyse SEMF and DZ10 nuclear
mass models. (Funny though: the DZ10 is not linear but a "DZ9" is.)

Apart from investigating different models, it is (likely) interesting to vary

model_discrepancy
    The value of model discrepancy is highly dependent on model/theory, but a
    good starting point is to use root-mean-square-deviation. But there are
    possibly "grooves" or values that "resonate" making the analysis behave in
    a peculiar way, unsure of the reason.

model_prior_type
    Two implemented: "uniform", flat, equal probability for all and "dilution"
    which uses the correlation between basis functions to lessen probability
    of models which are highly correlated.
    
    dilution_factor
        Used to increase or decrease the penalty of highly correlated models.
        |corr(phi)|^dilution_factor, typical values: 0.5, 1, 2

parameter_prior_type
    What type of parameter prior, only implemented "uniform", equal
    probability as inverse of the range of plausibility; "Gaussian", a
    multivariate normal distribution (MVN); "Zellner", MVN distribution
    using the basis functions to normalise the covariance matrix.
    
    parameter_prior_range
        Used with a "uniform" parameter prior, being the ranges of
        plausiblity. Might be estimated using theory.
        Can also be used for a truncated "Gaussian"?
    
    parameter_prior_mu
        Used with "Gaussian", being the expectation values of the parameters.
        A possiblity is computing these from a theoretical point of view;
        using results from other statistical methods is redundant.
        Can also be used with a "Zellner" prior for better results?
    
    parameter_prior_sigma
        Used with "Gaussian", being the covariance matrix of the parameters.
        Typical naive approach is a constant * identity matrix.
        Incompatible with "Zellner", see definition of Zellner g-prior.
        
    zellner_g
        Scaling factor to covariance matrix used in "Zellner".
        Def: p(alpha) = MVN(0, g [phi^T*phi]^-1)
        Possibly replace "0" with "mu" to improve "Zellner"?

------------------------------------------------------------------------------

TL;DR

model_prior_type: "uniform" or "dilution" -> dilution_factor: 0.5, 1, 2

model_discrepancy: your guess is as good as mine, units of MeV

parameter_prior_type: "uniform" or "Gaussian" or "Zellner"
    -> parameter_prior_range: for "uniform"
    -> parameter_prior_mu: for "Gaussian"
    -> parameter_prior_sigma: for "Gaussian"
    -> zellner_g: constant factor onto covariance matrix; 1 or len(D)

Learn by doing, try different stuff. Have fun.

'''

from model_class import Model


# ----------------------------------------------------------------------------
# VARIABLES / PARAMETERS
# ----------------------------------------------------------------------------

'''
MODEL VARIABLES
    model_name: what is the name of model/theory
    model_discrepancy: true nature = data + uncertainty = model + discrepancy
                       see Brynjarsdóttir and O'Hagan
                       doi: 10.1088/0266-5611/30/11/114007
    K: model dimension, number of basis functions
    model_prior_type: "uniform", "dilution"
    model_prior: unnecessary, will be computed at instantation
'''

model_name = 'SEMF'
model_discrepancy = 3.0  # MeV
model_prior_type = 'uniform'
dilution_factor = None
model_prior = None

'''
PARAMETER VARIABLES
    parameter_prior_type: "uniform", "Gaussian", "Zellner"
    parameter_prior_range: for "uniform" or maybe truncated "Gaussian"
    parameter_prior_mu: if "Gaussian" or "Zellner" expectation values
    parameter_prior_sigma: if "Gaussian" covariance matrix
    zellner_g: if "Zellner", scaling factor to covariance matrix
'''
parameter_prior_type = 'Zellner'
parameter_prior_range = None
parameter_prior_mu = None
parameter_prior_sigma = None
zellner_g = None

'''
Path to dataframe, this is expected to be a tab-separated file, it will be
loaded using pandas and ought to have columns according to
    N   Z   A   Element Ebinding    Eunc    "basis functions"
where "basis functions" can be any number of columns corresponding to a design
matrix, must start with "phi" for code to find it. Author used AME as data.
'''
dataframe_path = 'MASTERFRAMES/SEMF_ame16.tsv'

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------

model = Model(model_name=model_name,
              model_discrepancy=model_discrepancy,
              model_prior_type=model_prior_type,
              dilution_factor=dilution_factor,
              parameter_prior_type=parameter_prior_type,
              parameter_prior_range=parameter_prior_range,
              parameter_prior_mu=parameter_prior_mu,
              parameter_prior_sigma=parameter_prior_sigma,
              zellner_g=zellner_g,
              dataframe_path=dataframe_path)

model.analyse()
