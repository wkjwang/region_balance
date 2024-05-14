# -*- coding: utf-8 -*-
# FileName: F_statistic_bound
# Description: F_statistic_bound method to find varying Granger causal relations of time series
# Create by Amal in Jun 2016. Adjusting by Kaijun WANG in July, 2020


import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import lagmat2ds as lagmat2ds
from statsmodels.tools.tools import add_constant as add_constant
from statsmodels.regression.linear_model import OLS as OLS


def get_lagged_data(x, lag, addconst):
    x = np.asarray(x)
    # minimal data are needed to form time lag, and for degree of freedom in F_test
    num_lag = 2 * lag + int(addconst) + 1  # my adjusting from 3lag to 2lag
    if x.shape[0] < num_lag:
        raise ValueError("Mimimum {0} observations are needed for lag {1}".format(num_lag, lag))
    # create lagmat of both time series
    dta = lagmat2ds(x, lag, trim='both', dropex=1)
    # add constant
    if addconst:
        dtaown = add_constant(dta[:, 1:(lag + 1)], prepend=False, has_constant='add')
        dtajoint = add_constant(dta[:, 1:], prepend=False, has_constant='add')
    else:
        dtaown = dta[:, 1:(lag + 1)]
        dtajoint = dta[:, 1:]
    return dta, dtaown, dtajoint


def f_test(res2down, res2djoint, lag):
    result = {}
    res_df = res2djoint.df_resid
    ssr_own = res2down.ssr
    ssr_joint = res2djoint.ssr
    # Granger Causality test using ssr (F statistic)
    if lag <= 0:  # my checking zero denominator
        lag = 1
    if ssr_joint == 0:  # my checking zero denominator
        ssr_joint = 0.001
    fgc2 = (ssr_own - ssr_joint) / ssr_joint
    fgc1 = res_df * fgc2 / lag
    result['ssr_ftest'] = (fgc1,
                           stats.f.sf(fgc1, lag, res_df),
                           res_df, lag)
    return result


def fit_regression(dta, dtaown, dtajoint):
    # Run ols on both models without and with lags of second variable
    # OLS: Fit a linear model using Ordinary Least Squares
    res2down = OLS(dta[:, 0], dtaown).fit()
    res2djoint = OLS(dta[:, 0], dtajoint).fit()
    # for ssr based tests see:
    # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
    return res2down, res2djoint


def granger_tests(x, lag, addconst=True, do_lag=0, dta=0, dtaown=0, dtajoint=0):
    if do_lag:
        dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst)
    res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
    # my checking target variable that most elements are zeros, then no relation
    nzeros = findzeros_variable(dta)
    if nzeros >= 0.85:
        res2down.ssr = 0  # perfect self-regression, reduced_model
        res2djoint.ssr = 9  # bad relation between, full_model
    result = f_test(res2down, res2djoint, lag)
    p_value = result['ssr_ftest'][1]
    # if do_lag:
    #     return p_value
    return p_value, res2down, res2djoint


def findzeros_variable(data):
    nd = data.shape
    nzero = 0
    if len(nd) > 1:
        for i in range(nd[0]):
            if data[i, 0] == 0:
                nzero += 1
    else:
        for i in range(nd[0]):
            if data[i] == 0:
                nzero += 1
    nzero = nzero / nd[0]
    return nzero

'''
'''