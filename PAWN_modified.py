# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:21:15 2025

@author: TANGI

The script contains custom version of PAWN functions available at https://github.com/SAFEtoolbox/SAFE-python
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from safepython.util import empiricalcdf, split_sample
from safepython.lhcube import lhcube_shrink

from safepython.util import allrange, above, below
# Import SAFE modules:
from safepython import PAWN

#%%
def pawn_indices_mod(X, Y, n, Nboot=1, dummy=False, output_condition=allrange,
                 par=[]):

    """  Compute the PAWN sensitivity indices. The method was first introduced
    in Pianosi and Wagener (2015). Here indices are computed following the
    approximation strategy proposed by Pianosi and Wagener (2018), which can be
    applied to a generic input/output sample.

    The function splits the generic output sample to create the conditional
    output by calling internally the function PAWN.pawn_split_sample. The
    splitting strategy is an extension of the strategy for uniformy distributed
    inputs described in Pianosi and Wagener (2018) to handle inputs sampled
    from any distribution(see help of PAWN.pawn_split_sample for further explanation).

    Indices are then computed in two steps:
    1. compute the Kolmogorov-Smirnov (KS) statistic between the empirical
    unconditional CDF and the conditional CDFs for different conditioning
    intervals
    2. take a statistic (median, mean and max) of the results.

    Usage:

        KS_median, KS_mean, KS_max = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False)

        KS_median, KS_mean, KS_max, KS_dummy = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True)

    Input:
            X = set of inputs samples                      - numpy.ndarray(N,M)
            Y = set of output samples                      - numpy.ndarray(N, )
                                                        or - numpy.ndarray(N,1)
            n = number of conditioning intervals to
                assess the conditional CDFs:
                - integer if all inputs have the same number of groups
                - list of M integers otherwise

    Optional input:
        Nboot = number of bootstrap resamples to derive    - scalar
                confidence intervals
        dummy = if dummy is True, an articial input is     - boolean
                added to the set of inputs and the
                sensitivity indices are calculated for the
                dummy input.
                The sensitivity indices for the dummy
                input are estimates of the approximation
                error of the sensitivity indices and they
                can be used for screening, i.e. to
                separate influential and non-influential
                inputs as described in Khorashadi Zadeh
                et al. (2017)
                Default value: False
                (see (*) for further explanation).

    Output:
    KS_median = median KS across the conditioning      - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
      KS_mean = mean KS across the conditioning        - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
       KS_max = max KS across the conditioning         - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)

    Optional output (if dummy is True):
    KS_dummy = KS of dummy input (one value for       - numpy.ndarray(Nboot, )
                each bootstrap resample)

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    KS_median, KS_mean, KS_max = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False,
                      output_condition=allrange, par=[]))

    KS_median, KS_mean, KS_max, KS_dummy = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True,
                      output_condition=allrange, par=[]))

    Optional input:
    output_condition = condition on the output value to be     - function
                       used to calculate KS. Use the function:
                       - allrange to keep all output values
                       - below to consider only output
                          values below a threshold value
                          (Y <= Ythreshold)
                       - above to consider only output
                          values above a threshold value
                          (Y >= Ythreshold)
                    (functions allrange, below and above are defined in
                     safepython.util)
                 par = specify the input arguments of the      - list
                       'output_condition' function, i.e. the
                       threshold value when output_condition
                       is 'above' or 'below'.

    For more sophisticate conditions, the user can define its own function
    'output_condition' with the following structure:

        idx = output_condition(Y, param)

    where     Y = output samples (numpy.ndarray(N, ))
          param = parameters to define the condition (list of any size)
            idx = logical values, True if condition is satisfied, False
                  otherwise (numpy.ndarray(N, ))

    NOTE:
     (*) For screening influential and non-influential inputs, we recommend the
         use of the maximum KS across the conditioning intervals (i.e. output
         argument KS_max), and to compare KS_max with the index of the dummy
         input as in Khorashadi Zadeh et al. (2017).

    (**) For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    EXAMPLE:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from safepython.sampling import AAT_sampling
    from safepython.model_execution import model_execution
    from safepython import PAWN
    from safepython.plot_functions import boxplot1
    from safepython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 5000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Compute PAWN sensitivity indices:
    n = 10; # number of conditioning intervals
    KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n)
    plt.figure()
    plt.subplot(131); boxplot1(KS_median, Y_Label='KS (mean')
    plt.subplot(132); boxplot1(KS_mean, Y_Label='KS (mean')
    plt.subplot(133); boxplot1(KS_max, Y_Label='KS (max)')

    # Compute sensitivity indices for the dummy input as well:
    KS_median, KS_mean, KS_max, KS_dummy = PAWN.pawn_indices(X, Y, n, dummy=True)
    plt.figure()
    boxplot1(np.concatenate((KS_max, KS_dummy)),
             X_Labels=['X1', 'X2', 'X3', 'dummy'])

    REFERENCES

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

    Pianosi, F. and Wagener, T. (2015), A simple and efficient method
    for global sensitivity analysis based on cumulative distribution
    functions, Env. Mod. & Soft., 67, 1-11.

    REFERENCE FOR THE DUMMY PARAMETER:

    Khorashadi Zadeh et al. (2017), Comparison of variance-based and moment-
    independent global sensitivity analysis approaches by application to the
    SWAT model, Environmental Modelling & Software,91, 210-222.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info
    """
    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check other optional inputs
    ###########################################################################

    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 1:
        raise ValueError('"Nboot" must be >=1.')
    if not isinstance(dummy, bool):
        raise ValueError('"dummy" must be scalar and boolean.')
    if not callable(output_condition):
        raise ValueError('"output_condition" must be a function.')

    ###########################################################################
    # Compute indices
    ###########################################################################

    # Set points at which the CDFs will be evaluated:
    YF = np.linspace(min(Y), max(Y), 250)
    
    # Initialize sensitivity indices
    KS_median = np.nan * np.ones((Nboot, M))
    KS_mean = np.nan * np.ones((Nboot, M))
    KS_max = np.nan * np.ones((Nboot, M))
    if dummy: # Calculate index for the dummy input
        KS_dummy = np.nan * np.ones((Nboot, ))

    # Compute conditional CDFs
    # (bootstrapping is not used to assess conditional CDFs):
    FC = [np.nan] * M
    for i in range(M): # loop over inputs
        FC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            FC[i][k] = empiricalcdf(YY[i][k], YF)

    # Initialize unconditional CDFs:
    FU = [np.nan] * M

    # M unconditional CDFs are computed (one for each input), so that for
    # each input the conditional and unconditional CDFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output bootsize:
    bootsize = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # bootsize is equal to the sample size of the conditional outputs NC, or
    # its  minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    # help of the function).

    # To reduce the computational time (the calculation of empirical CDF is
    # costly), the unconditional CDF is computed only once for all inputs that
    # have the same value of bootsize[i].
    bootsize_unique = np.unique(bootsize)
    N_compute = len(bootsize_unique)  # number of unconditional CDFs that will
    # be computed for each bootstrap resample

    # Determine the sample size of the subsample for the dummy input.
    # The sensitivity
    # index for the dummy input will be estimated at this minimum sample size
    # so to estimate the 'worst' approximation error of the sensitivity index
    # across the inputs:
    if dummy:
        bootsize_min = min(bootsize) # we use the smaller sample size across
        # inputs, so that the sensitivity index for the dummy input estimates
        # the 'worst' approximation error of the sensitivity index across the
        # inputs:
        idx_bootsize_min = np.where([i == bootsize_min for i in bootsize])[0]
        idx_bootsize_min = idx_bootsize_min[0] # index of an input for which
        # the sample size of the unconditional sample is equal to bootsize_min

        if N_compute > 1:
            warn('The number of data points to estimate the conditional and '+
                 'unconditional output varies across the inputs. The CDFs ' +
                 'for the dummy input were computed using the minimum sample ' +
                 ' size to provide an estimate of the "worst" approximation' +
                 ' of the sensitivity indices across input.')

    # Compute sensitivity indices with bootstrapping
    for b in range(Nboot): # number of bootstrap resample

        # Compute empirical unconditional CDFs
        for kk in range(N_compute): # loop over the sizes of the unconditional output

            # Bootstrap resapling (Extract an unconditional sample of size
            # bootsize_unique[kk] by drawing data points from the full sample Y
            # without replacement
            idx_bootstrap = np.random.choice(np.arange(0, N, 1),
                                             size=(bootsize_unique[kk], ),
                                             replace='False')
            # Compute unconditional CDF:
            FUkk = empiricalcdf(Y[idx_bootstrap], YF)
            # Associate the FUkk to all inputs that require an unconditional
            # output of size bootsize_unique[kk]:
            idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
            for i in range(len(idx_input)):
                FU[idx_input[i]] = FUkk

        # Compute KS statistic between conditional and unconditional CDFs:
        KS_all = PAWN.pawn_ks(YF, FU, FC, output_condition, par)
        # KS_all is a list (M elements) and contains the value of the KS for
        # for each input and each conditioning interval. KS[i] contains values
        # for the i-th input and the n_eff[i] conditioning intervals, and it
        # is a numpy.ndarray of shape (n_eff[i], ).

        #  Take a statistic of KS across the conditioning intervals:
        KS_median[b, :] = np.array([np.median(j) for j in KS_all])  # shape (M,)
        KS_mean[b, :] = np.array([np.mean(j) for j in KS_all])  # shape (M,)
        KS_max[b, :] = np.array([np.max(j) for j in KS_all])  # shape (M,)

        if dummy:
            # Compute KS statistic for dummy parameter:
            # Bootstrap again from unconditional sample (the size of the
            # resample is equal to bootsize_min):
            idx_dummy = np.random.choice(np.arange(0, N, 1),
                                         size=(bootsize_min, ),
                                         replace='False')
            # Compute empirical CDFs for the dummy input:
            FC_dummy = empiricalcdf(Y[idx_dummy], YF)
            # Compute KS stastic for the dummy input:
            KS_dummy[b] = PAWN.pawn_ks(YF, [FU[idx_bootsize_min]], [[FC_dummy]],
                                  output_condition, par)[0][0]

    if Nboot == 1:
        KS_median = KS_median.flatten()
        KS_mean = KS_mean.flatten()
        KS_max = KS_max.flatten()

    if dummy:
        return KS_median, KS_mean, KS_max, KS_dummy
    else:
        return KS_median, KS_mean, KS_max
    
    
#%%

def pawn_plot_cdf_mod(X, Y, n, n_col=5, Y_Label='', cbar=False,
                  labelinput=''):

    """ This function computes and plots the unconditional output Cumulative
    Distribution Funtions (i.e. when all inputs vary) and the conditional CDFs
    (when one input is fixed to a given conditioning interval, while the other
    inputs vary freely).

    The function splits the output sample to create the conditional output
    by calling internally the function PAWN.pawn_split_sample. The splitting
    strategy is an extension of the strategy for uniformy distributed inputs
    described in Pianosi and Wagener (2018) to handle inputs sampled from any
    distribution.
    (see help of PAWN.pawn_split_sample for further explanation).

    The sensitivity indices for the PAWN method (KS statistic) measures the
    distance between these conditional and unconditional output CDFs
    (see help of PAWN.pawn_indices for further details and reference).

    Usage:
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, n_col=5, Y_Label='output y',
                                        cbar=False, labelinput='')

    Input:
             X = set of inputs samples                     - numpy.ndarray(N,M)
             Y = set of output samples                     - numpy.ndarray(N,)
                                                        or - numpy.ndarray(N,1)
             n = number of conditioning intervals
                 - integer if all inputs have the same number of groups
                 - list of M integers otherwise

    Optional input:
         n_col = number of panels per row in the plot      - integer
                 (default: min(5, M))
       Y_Label = legend for the horizontal axis            - string
                 (default: 'output y')
          cbar = flag to add a colobar that indicates the  - boolean
                 centers of the conditioning intervals for
                 the different conditional CDFs:
                 - if True = colorbar
                 - if False = no colorbar
    labelinput = label for the axis of colorbar (input    - list (M elements)
                 name) (default: ['X1','X2',...,XM'])

    Output:
            YF = values of Y at which the CDFs FU and FC   - numpy.ndarray(P, )
                 are given
            FU = values of the empirical unconditional     - list(M elements)
                 output CDFs. FU[i] is a numpy.ndarray(P, )
                 (see the Note below for further
                 explanation)
            FC = values of the empirical conditional       - list(M elements)
                 output CDFs for each input and each
                 conditioning interval.
                 FC[i] is a list of n_eff[i] CDFs
                 conditional to the i-th input.
                 FC[i][k] is obtained by fixing the i-th
                 input to its k-th conditioning interval
                 (while the other inputs vary freely),
                 and it is a np.ndarray of shape (P, ).
                 (see the Note below for further
                 explanation)
           xc = subsamples' centers (i.e. mean value of    - list(M elements)
                Xi over each conditioning interval)
                xc[i] is a np.ndarray(n_eff[i],) and
                contains the centers for the n_eff[i]
                conditioning intervals for the i-th input.

    Note:
    (*)  For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    (**) FU[i] and FC[i][k] (for any given i and k) are built using the same
         number of data points so that two CDFs can be compared by calculating
         the KS statistic (see help of PAWN.pawn_ks and PAWN.pawn_indices
         for further explanation on the calculation of the KS statistic).

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from safepython.sampling import AAT_sampling
    from safepython.model_execution import model_execution
    from safepython import PAWN
    from safepython.plot_functions import boxplot1
    from safepython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 1000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Calculate and plot CDFs
    n = 10 # number of conditioning intervals
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True) # Add colorbar

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 17} # font
    
    # Options for the graphic
    tickfont = {'fontname': 'DejaVu Sans', 'fontsize': 12} # font
    
    # Options for the graphic
    clbfont = {'fontname': 'DejaVu Sans', 'fontsize': 12} # font
    
    colorscale = 'gray' # colorscale
    # Text formating of ticklabels
    yticklabels_form = '%3.1f' # float with 1 character after decimal point
    # yticklabels_form = '%d' # integer

    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check optional inputs for plotting
    ###########################################################################

    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')
    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')
    if not isinstance(cbar, bool):
        raise ValueError('"cbar" must be scalar and boolean.')
    if not labelinput:
        labelinput = [np.nan]*M
        for i in range(M):
            labelinput[i] = 'X' + str(i+1)
    else:
        if not isinstance(labelinput, list):
            raise ValueError('"labelinput" must be a list with M elements.')
        if not all(isinstance(i, str) for i in labelinput):
            raise ValueError('Elements in "labelinput" must be strings.')
        if len(labelinput) != M:
            raise ValueError('"labelinput" must have M elements.')

    ###########################################################################
    # Compute CDFs
    ###########################################################################

    # Set points at which the CDFs will be evaluated:
    YF = np.linspace(min(Y), max(Y), 1000)

    # Compute conditional CDFs:
    FC = [np.nan] * M
    for i in range(M): # loop over inputs
        FC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            FC[i][k] = empiricalcdf(YY[i][k], YF)

    # Initialize unconditional CDFs:
    FU = [np.nan] * M

    # M unconditional CDFs are computed (one for each input), so that for
    # each input the conditional and unconditional CDFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output NU:
    NU = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # NU is equal to the sample size of the conditional outputs NC, or its
    # minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    #  help of the function).

    # To reduce the computational time (the calculation of empirical CDF is
    # costly), the unconditional CDF is computed only once for all inputs that
    # have the same value of NU[i].
    NU_unique = np.unique(NU)
    N_compute = len(NU_unique) # number of unconditional CDFs that will be computed

    for kk in range(N_compute): # loop over the sizes of the unconditional output

        # Extract an unconditional sample of size NU_unique[kk] by drawing data
        # points from the full sample Y without replacement
        idx = np.random.choice(np.arange(0, N, 1), size=(NU_unique[kk], ),
                               replace='False')
        # Compute unconditional output CDF:
        FUkk = empiricalcdf(Y[idx], YF)
        # Associate the FUkk to all inputs that require an unconditional output
        # of size NU_unique[kk]:
        idx_input = np.where([i == NU_unique[kk] for i in NU])[0]
        for j in range(len(idx_input)):
            FU[idx_input[j]] = FUkk

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    plt.figure()

    for i in range(M): # loop over inputs

        # Prepare color and legend
        cmap = mpl.cm.get_cmap(colorscale, n_eff[i]+1) # return colormap,
        # (n+1) so that the last line is not white
        # Make sure that subsample centers are ordered:
        iii = np.argsort(xc[i])
        ccc = np.sort(xc[i])

        plt.subplot(n_row, n_col, i+1)
        ax = plt.gca()

        if cbar: # plot dummy mappable to generate the colorbar
            Map = plt.imshow(np.array([[0, 1]]), cmap=cmap)
            plt.cla() # clear axes (do not display the dummy map)

        # Plot a horizontal dashed line at F=1:
        plt.plot(YF, np.ones(YF.shape), '--k')

        # Plot conditional CDFs in gray scale:
        for k in range(n_eff[i]):
            plt.plot(YF, FC[i][iii[k]], color=cmap(k), linewidth=2)
        plt.xticks(**tickfont); plt.yticks(**tickfont)
        plt.xlabel(Y_Label, **clbfont)

        # #plot y label with uncertanty variable name
        # plt.ylabel(labelinput[i], **pltfont)

        # Plot unconditional CDF in red:
        plt.plot(YF, FU[i], 'r', linewidth=3)

        plt.xlim([min(YF), max(YF)]); plt.ylim([-0.02, 1.02])

        if cbar:  
            cb_ticks = [' '] * n_eff[i]
            for k in range(n_eff[i]):
                if abs(ccc[k]) >= 1000:
                    cb_ticks[k] = f"{ccc[k]:.1e}"  # Scientific notation
                else:
                    cb_ticks[k] = f"{ccc[k]:0.3g}"  # 4 significant digits
        
            cb = plt.colorbar(Map, ax=ax,
                              boundaries=np.linspace(0, 1 - 1 / (n_eff[i] + 1),
                                                     n_eff[i] + 1))
            cb.set_label(labelinput[i], **pltfont)
            cb.set_ticks(np.linspace(1 / (2 * (n_eff[i] + 1)), 
                                     1 - 3 / (2 * (n_eff[i] + 1)), 
                                     n_eff[i]))
            cb.set_ticklabels(cb_ticks, **clbfont)
            cb.ax.tick_params(labelsize=clbfont['fontsize'])

            # Map.set_clim(0,1-1/(n+1))
            ax.set_aspect('auto') # Ensure that axes do not shrink

    # plt.suptitle(Y_Label, **pltfont)

    return YF, FU, FC, xc
