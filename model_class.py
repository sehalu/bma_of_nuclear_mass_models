# -*- coding: utf-8 -*-
'''
Created on Thu Apr  9 12:26:02 2020

@author: sehalu

Work in progress, might never be used.
Experimental code to use a class type instead of using a single
variable and a myriad of functions. Of course there are still many methods,
but the aim of this code is to make it more human readable.
'''

# Standard libraries
import numpy as np
from numpy.linalg import pinv
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from time import time, sleep

import sys, os, platform

# Own modules
from bma import evidences
from bma import posterior_model_probability as pmp


# =============================================================================
# Main parameter class, p(alpha)
# =============================================================================

class Model():

    # Functions/methods are in seperate files
    from _model_fncs import get_model, \
                            model_prior_uniform, \
                            model_prior_dilution
    from _parameter_fncs import zellner_g_prior

    # Important variables
    model_name = None
    model_discrepancy = None

    data_set = None
    data = None
    uncertainty = None
    phi = None
    E = None

    # parameter stuff
    nbr_of_parameters = None
    parameters = None
    parameter_prior_type = None
    parameter_prior_range = None
    parameter_prior_mu = None
    parameter_prior_sigma = None
    zellner_g = None

    # model stuff
    K = None
    model_prior_type = None
    dilution_factor = None
    model_prior = None
    nbr_of_models = None

    # ========================================================================
    # INITIALISE
    def __init__(self,
                 # data stuff
                 data: pd.DataFrame=None,
                 design_matrix: pd.DataFrame=None,
                 # parameter stuff
                 nbr_of_parameters: int=None,
                 parameters: list=None,
                 parameter_prior_type: str=None,
                 parameter_prior_range=None,
                 parameter_prior_mu: np.ndarray=None,
                 parameter_prior_sigma:np.ndarray=None,
                 zellner_g: int=None,
                 # model stuff
                 model_name: str=None,
                 model_discrepancy: float=None,
                 K: int=None,
                 model_prior_type: str=None,
                 model_prior: list=None,
                 dilution_factor: float=None,
                 # or with file
                 initfile: str=None,
                 dataframe_path: str=None
                 ):

        if initfile:
            self.assign_with_file(initfile)

        if dataframe_path:
            self.assign_dataframe(dataframe_path)
            self.data_set = dataframe_path

        # =====================================================================
        # Parameter __init__

        if nbr_of_parameters is not None:
            assert type(nbr_of_parameters) is int, ('Number of parameters '
                                                    + 'must be integer!')
            self.nbr_of_parameters = nbr_of_parameters
            self.nbr_of_models = 2**nbr_of_parameters - 1

        if parameters is not None:
            self.parameters = np.atleast_1d(parameters)
        if self.nbr_of_parameters is None:
            self.nbr_of_parameters = len(parameters)

        if parameter_prior_type is not None:
            if parameter_prior_type.lower() in ['uniform', 'flat']:
                self.parameter_prior_type = 'uniform'
            elif parameter_prior_type.lower() in ['gaussian', 'normal']:
                self.parameter_prior_type = 'Gaussian'
            elif parameter_prior_type.lower() in ['zellner', 'zellner_g']:
                self.parameter_prior_type = 'Zellner'

        if parameter_prior_range is not None:

            # Make sure priors have correct shape: (2 rows, K columns)
            # parameter_prior_range[0] is lower limit,
            # parameter_prior_range[1] is upper limit
            parameter_prior_range = np.atleast_2d(parameter_prior_range)
            ps = parameter_prior_range.shape
            assert self.nbr_of_parameters in ps, ('Ranges and number of '
                                                  + 'parameters do not match')
            assert 2 in ps, 'Must have both lower and upper bounds'
            if ps == (self.nbr_of_parameters, 2):
                parameter_prior_range = parameter_prior_range.T
            self.parameter_prior_range = parameter_prior_range

        if parameter_prior_mu is not None:
            assert len(parameter_prior_mu)==self.nbr_of_parameters, 'Mean values length does not match number of parameters'
            self.parameter_prior_mu = np.atleast_1d(parameter_prior_mu)

        if parameter_prior_sigma is not None:
            assert len(parameter_prior_sigma) == len(parameter_prior_sigma[0]), 'Sigma must be a square matrix'
            self.parameter_prior_sigma = np.atleast_2d(parameter_prior_sigma)

        if self.parameter_prior_type == 'Zellner':
            if zellner_g is not None:
                self.zellner_g = zellner_g
            self.assign_zellner_g()

        # =====================================================================
        # Model __init__

        if model_name is not None:
            self.model_name = model_name

        if model_discrepancy is not None:
            self.model_discrepancy = model_discrepancy

        if K is not None:
            self.K = K

        if model_prior_type is None:
            model_prior_type = self.model_prior_type

        if model_prior_type is not None:
            if model_prior_type.lower() in ['uniform', 'flat']:
                self.model_prior_type = 'uniform'
                self.model_prior = self.model_prior_uniform()
            elif model_prior_type.lower() in ['dilute', 'dilution']:
                self.model_prior_type = 'dilution'
                self.dilution_factor = dilution_factor
                self.model_prior = self.model_prior_dilution()
            else:
                print('Wrong type of model given!')
    # END INITIALISE
    # ========================================================================

    def __str__(self):
        '''
        For printing information about model under investigation / run
        '''
        information = ''

        if self.model_name is not None:
            information += f'Investigating "{self.model_name}"\n'
        if self.data_set is not None:
            information += f'Using data set "{self.data_set}"\n'
        if self.nbr_of_parameters is not None:
            information += f'Number of parameters = {self.nbr_of_parameters}\n'
        if self.parameters is not None:
            information += f'Parameters are {self.parameters}\n'
        if self.parameter_prior_type is not None:
            information += f'Using a {self.parameter_prior_type} ' \
                           + 'parameter prior'
            if self.parameter_prior_type == 'Zellner':
                information += f' with g = {self.zellner_g}\n'
            else:
                information += '\n'
        if self.parameter_prior_range is not None:
            information += 'Parameter values within ' \
                           + f'{self.parameter_prior_range}\n'
        if self.parameter_prior_mu is not None:
            information += 'Expected parameter values = ' \
                           + f'{self.parameter_prior_mu}\n'
        if self.parameter_prior_sigma is not None:
            information += 'Parameter covariance matrix = ' \
                           + f'{self.parameter_prior_sigma}\n'
        if self.model_prior_type is not None:
            information += f'Using a {self.model_prior_type} model prior\n'

        if information == '':
            information = 'No information available! Have you really set up?\n'

        return information

    def info(self, print_or_write: str=None, outfile: str=None):
        '''
        Print or write information about variables
        '''
        information = self.__str__()

        if print_or_write is None or print_or_write == 'print':
            print('\n' + information)
        elif print_or_write == 'write':
            outfile = outfile if outfile is not None else 'out/information.txt'
            print(f'Writing information to "{outfile}" ... ', end='')
            with open(outfile, 'w') as f:
                f.write(information)
            print('Done!\n')

    def assign_with_file(self, filename: str):
        '''
        Ask for what all variables ought to be / assign them all.
        '''
        assert os.path.exists(filename), 'Initfile does not exist!'

        with open(filename) as f:
            # for every line
            for line in f.readlines():
                # if line assigns appropriate variable
                if line.split('=')[0].strip(' ') in dir(self):
                    exec('self.'+line)  # ex: model_name = 'DZ10'

    def assign_dataframe(self, dataframe_path: str):

        assert os.path.exists(dataframe_path), 'Dataframe does not exist!'
        df = pd.read_csv(dataframe_path, sep='\t')

        try:
            self.data = df['Ebinding']
        except KeyError:
            print('Cannot assign data!\nNo entry "Ebinding"!')

        try:
            self.uncertainty = df['Eunc']
        except KeyError:
            print('Cannot assign uncertainties!\nNo entry "Eunc"')

        try:
            self.parameters = np.array([c for c in df.columns
                                        if c.startswith('phi')])
            if self.nbr_of_parameters is None:
                self.nbr_of_parameters = len(self.parameters)
            self.phi = df[self.parameters]
        except KeyError:
            print('Cannot assign designmatrix!\n',
                  'No entries starting with "phi"')

        try:
            if self.parameters is not None:
                if self.K is None:
                    self.K = len(self.parameters)
                self.nbr_of_models = 2**self.K - 1
        except AttributeError:
            print('Model has no parameters!')

        if self.model_discrepancy is None:
            self.E = self.uncertainty**2
        else:
            self.E = self.uncertainty**2 + self.model_discrepancy**2

    def assign_zellner_g(self):
        try:
            assert self.phi is not None, ('Cannot assign Zellner g!\n'
                                          + 'No design matrix.')
        except AttributeError:
            print('Cannot assign Zellner g!\n No design matrix.')

        m, s = self.zellner_g_prior()
        if self.parameter_prior_mu is None:
            self.parameter_prior_mu = m
        self.parameter_prior_sigma = s

    def assign_model_prior(self, prior_type: str=None,
                           dilution_factor: float=None):

        if prior_type is None:
            assert self.model_prior_type is not None, ('No model prior type '
                                                       + 'specified!')
        else:
            if prior_type.lower() in ['uniform', 'flat']:
                self.model_prior_type = 'uniform'
            elif prior_type.lower() in ['dilution', 'dilute']:
                self.model_prior_type = 'dilution'
            else:
                raise NotImplementedError('Only "uniform" and "flat" model '
                                          + 'priors implemented')

        if dilution_factor is not None:
            self.dilution_factor = dilution_factor

        if self.nbr_of_models is None:
            assert self.K is not None, 'Cannot assign model prior if no "K"'
            self.nbr_of_models = 2**self.K - 1

        if self.model_prior_type == 'uniform':
            self.model_prior = self.model_prior_uniform()
        elif self.model_prior_type == 'dilution':
            if self.dilution_factor is None:
                print('Setting dilution factor to "1.0"')
                self.dilution_factor = 1.0
            self.model_prior = self.model_prior_dilution()

    # =========================================================================
    # WRITE POSTERIOR DISTRIBUTIONS TO FILE
    # =========================================================================

    def write_model_posterior(self, log_pmp: pd.DataFrame,
                              output_dir: str=None, filename: str=None):

        # Use data set filename as basis for output filename
        if filename is None:
            filename = self.data_set.split('/')[-1].split('\\')[-1]
            filename = filename.split('.')[0]

        # If no output directory given write to "out/" in current directory
        if output_dir is None:
            output_dir = 'out/'

        os.mkdir(output_dir) if not os.path.exists(output_dir) else ''
        filename = output_dir + filename + '_model_posterior.out'

        print(f'Writing model posterior to "{filename}" ... ', end='')

        log_pmp.to_csv(filename, sep='\t', index=False)

        print('Done!\n')

    def write_parameter_posterior(self, parameter_posterior: np.ndarray,
                                  output_dir: str=None, filename: str=None):

        # Use data set filename as basis for output filename
        if filename is None:
            filename = self.data_set.split('/')[-1].split('\\')[-1]
            filename = filename.split('.')[0]

        # If no output directory given write to "out/" in current directory
        if output_dir is None:
            output_dir = 'out/'

        os.mkdir(output_dir) if not os.path.exists(output_dir) else ''
        filename = output_dir + filename + '_parameter_posterior.out'

        print(f'Writing parameter posterior to "{filename}" ... ', end='')

        # Print parameter posterior, mean and covariance, for every model
        # For many models this should probably be omitted
        with open(filename, 'w') as f:
            f.write('# Model number\nParameter\tmu\tCovariance matrix\n')
            for k in range(1, self.nbr_of_models+1):
                k_bool = np.array([int(i) for i in f'{k:0{self.K}b}'],
                                  dtype=bool)
                model_k = self.parameters[k_bool]
                f.write(f'# k={k}, {model_k}\n')
                mu = parameter_posterior[k-1]['post_mu']
                Sigma = parameter_posterior[k-1]['post_Sigma']
                Sigstr = ['\t'.join([str(s) for s in S]) for S in Sigma]
                for i in range(len(mu)):
                    f.write(f'{model_k[i]}\t{mu[i]}\t{Sigstr[i]}\n')
                f.write(f'{"".join(["#"]*79)}\n')

        print('Done!\n')

    def save(self, output_dir: str=None, pmp_filename: str=None,
             ppp_filename: str=None):

        if self.model_posterior is not None:
            self.write_model_posterior(self.model_posterior,
                                       output_dir=output_dir,
                                       filename=pmp_filename)
        if self.parameter_posterior is not None:
            self.write_parameter_posterior(self.parameter_posterior,
                                           output_dir=output_dir,
                                           filename=ppp_filename)

    # =========================================================================
    # CALCULATE EVIDENCES
    # =========================================================================

    def check(self):
        '''
        Silly check to make sure all necessary variables are set for
        calculating evidences, posterior parameter probabilites and
        posterior model probabilites.
        '''
        errors = 0
        try:
            self.data
        except AttributeError:
            print('We need data to analyse!')
            errors += 1
        try:
            self.E
        except AttributeError:
            print('We need uncertainty to analyse!')
            errors += 1
        try:
            self.parameter_prior_type
        except AttributeError:
            print('We need parameter prior type to analyse!')
            errors += 1
        try:
            self.nbr_of_models
        except AttributeError:
            print('We need number of models to analyse!')
            errors += 1

        if errors > 0:
            raise AttributeError('Fix variables before performing analysis!')

    def calculate_evidences(self):
        '''
        Calculate evidences:
            p(D|M_k) = Integral[ p(alpha_k) d alpha_k p(D|M_k,alpha_k) ]
        The function this method calls can be swapped for another.
        By author an analytical solution is implemented, since it is possible
        when investigating linear models using uniform or multivariate normal
        parameter prior distributions. For implementing analysis of non-linear
        models (or for some other reasons) a numerical solver is possible.
        '''
        self.check()

        # Using local variables might improve speed
        nbr_of_models = self.nbr_of_models
        D = self.data
        E = self.E
        pp_type = self.parameter_prior_type
        K = self.K
        parameters = self.parameters
        phi = self.phi
        prior_type = self.parameter_prior_type
        prior_values = (self.parameter_prior_range,
                        self.parameter_prior_mu,
                        self.parameter_prior_sigma)

        t0 = time()  # Time it for fun

        # Initialise vectors
        log_evidence = np.empty(shape=(nbr_of_models,), dtype=float)
        ppp = np.empty(shape=(nbr_of_models,), dtype=object)

        print('\nCalculating model evidences')

        for k in range(1, nbr_of_models+1):
            print(f'Calculating evidence for model {k} of {nbr_of_models}, ',
                  end='')
            phi_k, pp_k = self.get_model(k, K=K, parameters=parameters,
                                         phi=phi, with_prior=True,
                                         prior_type=prior_type,
                                         prior_values=prior_values)
            log_evidence[k-1], ppp[k-1] = evidences(D, E, phi_k, pp_type, pp_k)
            print(f'{int(100*k/nbr_of_models)}% completed')

        print('Evidences calculated, 100% completed\n')

        # Look at clock, print time
        tf = time() - t0
        print(f'Calculations took {tf:.2f} seconds')
        print('Time per model evidence calculation: '
              f'{tf/self.nbr_of_models:.2f} s/model\n')

        # Save evidences and posterior parameter probability as instance
        # variables. Very slow?
        self.log_evidence = log_evidence
        self.parameter_posterior = ppp
        sleep(3)

    def calculate_posterior_model_probability(self):
        '''
        Compute posterior model proability:
            p(M_k|D) = p(D|M_k) p(M_k) / p(D)
        where the total evidence is simply a normalising factor
            p(D) = Sum[ p(M_k|D) p(M_k) ]
        This function can be replaced to implement e.g. numerical methods.
        '''
        if self.model_prior is None:
            msg = '\nNo model prior given! Fixing it using '
            msg += f'{self.model_prior_type} prior '
            if self.model_prior_type == 'dilution':
                msg += f'and {self.dilution_factor}'
            print(msg+'...', end='')
            self.assign_model_prior()
            print('Done!')

        print('\nCalculating model posterior probability ... ', end='')
        model_posterior = pmp(self.log_evidence, self.model_prior)
        print('Done!\n')

        # Save posterior model probability distribution as instance variable
        self.model_posterior = model_posterior

    def analyse(self):
        '''
        Calculate posterior model and parameter distributions based on some
        data, i.e. performing Bayesian analysis.
        '''

        # Clear terminal before writing anything
        if platform.system() == 'Windows':
            os.system('CLS')
        elif platform.system() == 'Linux':
            os.system('clear')

        # Start with describing what we are doing
        self.info()
        print('='*40+'\n')
        sleep(5)

        self.calculate_evidences()
        self.calculate_posterior_model_probability()

        print('Done with analysing!\n', '='*40, '\n\n')
