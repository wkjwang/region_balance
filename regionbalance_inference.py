# -*- coding: utf-8 -*-
# Created by Kaijun WANG, Zhongjian Miao in May-July, 2020, adjusted in May 2024

import math
from regionbalance_test_functions import *
from statsmodels.tsa.stattools import adfuller


def adf_test(data, lag=0, s_thres=0.05):
    ns = data.shape
    na = np.zeros(ns[1],dtype=int)
    lag_self = np.zeros([4, ns[1]],dtype=int)
    for i in range(ns[1]):
        if lag: rad = adfuller(data[:,i], lag)
        else:
            rad = adfuller(data[:,i], autolag='AIC')
            lag_self[0, i] = rad[2]
        if rad[1] >= s_thres:
            na[i] += 1
            n1 = ns[0]//3
            n2 = ns[0] - n1
            rad1 = adfuller(data[0:n1, i], autolag='AIC')
            lag_self[1, i] = rad1[2]
            if rad1[1] >= s_thres: na[i] += 1
            rad1 = adfuller(data[n1:n2, i], autolag='AIC')
            lag_self[2, i] = rad1[2]
            if rad1[1] >= s_thres: na[i] += 1
            rad1 = adfuller(data[n2:ns[0], i], autolag='AIC')
            lag_self[3, i] = rad1[2]
            if rad1[1] >= s_thres: na[i] += 1
    return na, lag_self


def test_relation_series(da_yx, da_xy, lag, signif_thres, lag_add=1):
    # test the relation between two time series X & Y
    # result: 1 for relation X->Y, -1 for Y->X, 0 for no relation
    zone_value = 0  # no relation
    p_value1, res_yx_y, res_yx_joint = granger_tests(da_yx, lag, True, lag_add)
    if p_value1 < signif_thres and p_value1 != -1:  # relation direction X->Y
        zone_value = 1
    else:  # Y->X
        p_value1, res_xy_x, res_xy_joint = granger_tests(da_xy, lag, True, lag_add)
        if p_value1 < signif_thres and p_value1 != -1:
            zone_value = -1
    return zone_value


def differ_series(i_data, i_start, i_end, i_mode=3):
    # single- and double-interval differences of time series
    i_end = i_end - i_start + 1
    i_start = 0
    avg_sf = -1
    avg_df = -1
    # i_mode == 3: 'single' + 'double'
    if i_mode == 3 or i_mode == 1:  # 'single'
        num = i_end - 1
        diff1 = i_data[i_start:i_end-1]
        diff2 = i_data[i_start+1:i_end]
        sdiff = np.abs(diff2 - diff1)
        avg_sf = sdiff.sum() / num
    if i_mode == 3 or i_mode == 2:  # 'double'
        diff1 = np.diff(i_data[i_start:i_end:2])
        diff1 = np.abs(diff1)
        num1 = len(diff1)
        diff2 = np.diff(i_data[i_start+1:i_end:2])
        diff2 = np.abs(diff2)
        num2 = len(diff2)
        avg_df = (diff1.sum() + diff2.sum()) / (num1+num2)
    return avg_sf, avg_df


def variation_bound(dat_yx, signif_thres, lag, wide, step_slide=0, mode=3):
    # three sub-branches of our variation-bound method:
    # mode = 1: 'single' differences of time series, 2: 'double', 3: both 1 & 2

    # time series with lags, degree of freedom in F_test, need min window wide
    n_sample = len(dat_yx[:, 0])
    num_lag = 2 * lag + 1
    if wide <= num_lag or n_sample < wide + lag:
        raise KeyError('it needs min window wide > 2*lag + 1')

    # (1) ================== initialization ==================
    dat_xy = dat_yx.copy()
    dat_xy[:, 0] = dat_yx[:, 1]
    dat_xy[:, 1] = dat_yx[:, 0]

    # record of causal-relation results
    q_result = dict()

    # parameters depend on parameter 'wide' (i.e., length of sliding window)
    if wide < 36:
        u_wide = round(wide/2)  # length of variation-checking segment
    else:
        u_wide = 1 + 3 * round(math.sqrt(wide))
    du = u_wide//2  # moving step in variation-checking segment
    if du < 1:
        du = 1
    n_work = u_wide//du  # working times for testing
    du2 = du//2
    if n_work < 2:
        n_work = 2
    n_test = n_work
    n_check = n_work
    count_slide = 0  # count for sliding times of sliding-window
    count_exceed = 0  # count for neighbour segments inside variation bounds
    mp = 0.95  # a bound is multiplied by mp
    if not step_slide:
        step_slide = du  # sliding step of window: u_wide

    # turns on/off our variation-bound method, >3: our_method_off
    if mode > 3:
        n_test = 0
        n_check = 0
        u_wide = 0

    # array indices i & j are used for accurate data amount
    i = 0  # sliding-window start
    j = wide - 1  # sliding-window end
    j_end = j + u_wide  # variation-checking end
    n_stop = 0
    i_j_jump = 1
    count_test = n_test  # count for a loop

    while n_stop < 2:
        dt = 0
        front_jump = 0
        count_check = 0
        while count_check < n_check:
            # (2)========== Computation of variation bounds ==========
            # variation bounds of target variable in sliding-window [i,j]
            dt = -du
            if count_check == 0:
                sf_max, df_max = differ_series(dat_yx[i:j+1,0], i, j, 3)
                sf_max = mp * sf_max
                df_max = mp * df_max
                dt2 = 0
            # (3)===== Variation evaluation in neighbour segments =====
            # front neighbour segment [j,j+dt] with start j
            dt2 = dt2 + du
            j_tail = j + dt2
            if count_check + 1 == n_check:
                j_tail = j_end
            sf_jdt, df_jdt = differ_series(dat_yx[j:j_tail+1, 0], j, j_tail, 3)
            # checking whether variation bounds are exceeded in neighbour segments
            if sf_jdt > sf_max or df_jdt > df_max:   # 'single' or 'double'
                count_test = 0
                front_jump += 1
                if front_jump == 1:
                    count_exceed += 1
            elif count_check == n_check - 1:
                count_test = 0
            count_check += 1

        # (4-5)============== Granger causal test ================
        # Granger test in segments [i,j], [i,j+dt], [i,j+u_wide]
        if not i_j_jump and front_jump:
            dt = -u_wide
        while count_test <= n_test:
            # the first/second loop is for segment [i+dt,j]
            if count_test <= 1:
                j_head = i + dt
                j_tail = j
                if j_head < 0:
                    j_head = 0
            else:
                # the third loop for segment [i,j+dt] covers sliding-window & front segment
                j_head = i
                j_tail = j + dt
                if count_test == n_test:  # for the last loop
                    j_tail = j_end
            if j_tail > j_end:
                j_tail = j_end
            # more data points are added to keep window [i, j] after data cutting
            if j_head > lag:
                j_head_lag = j_head - lag
            else:
                j_head_lag = 0
            # Granger causal test for relation X->Y and Y->X
            avalue = test_relation_series(dat_yx[j_head_lag:j_tail + 1, :], dat_xy[j_head_lag:j_tail + 1, :], lag, signif_thres, 1)
            # avalue: 1 for relation X->Y, -1 for Y->X, 0 for no relation
            q_result[(j_head, j_tail)] = avalue
            dt += du
            count_test += 1

        # (6)========== Sliding to next window or stop ===========
        # resetting to default once test
        count_test = n_test
        count_slide += 1
        i_j_jump = front_jump

        i = i + step_slide
        j = i + wide - 1
        j_end = j + u_wide
        if j >= n_sample - 1:
            j = n_sample - 1
            i = j - wide + 1
            j_end = n_sample - 1
            n_check = 0  # skip variation evaluation
            n_stop += 1
        elif j_end >= n_sample:
            j_end = n_sample - 1
            n_stop += 1
            if j_end - j < du:
                n_check = 0  # skip variation evaluation
                if mode <= 3:
                    count_test = 1
            elif j_end - j <= du + du2:
                n_check = n_check - 1

    return q_result, count_slide - count_exceed, count_slide
