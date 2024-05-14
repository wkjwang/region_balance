# -*- coding: utf-8 -*-
# Created by Kaijun WANG in August 2020, adjusted in May 2024

import numpy as np
from regionbalance_inference import variation_bound, adf_test
from regionbalance_functions import syntheticdata_generate, counts2label, accuracy_rate, printresult
import timeit


n_iteration = 10  # repetition number
window_size = [20, 30, 40]
step_slide = [5, 10]
std_noise = 0.45  # 0.1 0.45 0.7
time_lag = 2
signif_thres = 0.05  # F test threshold
nsample = 1000  # length of time series
da_head = ['target', 'feature']

window_min = 2 * time_lag + 2
n_window_size = len(window_size)
n_step = len(step_slide)
acy_balance = np.zeros((n_step, n_window_size), dtype=float)
time_balance = np.zeros((n_step, n_window_size), dtype=float)
inbound_rate = np.zeros((n_step, n_window_size), dtype=float)
acy_slide = np.zeros((n_step, n_window_size), dtype=float)
time_slide = np.zeros((n_step, n_window_size), dtype=float)

nk = 0
n_total = n_iteration * n_window_size * n_step
time1 = timeit.default_timer()
for k in range(n_iteration):
    data2_yx, clabels, da_head = syntheticdata_generate(std_noise, time_lag)
    rad, lagself = adf_test(data2_yx)
    if rad[0] > 3 and rad[1] > 3:
        print('not stationary! linear Granger causal test might be inaccurate !')
    elif rad[0] < 1 and rad[1] < 1:
        print('all series are stationary !')
    else:
        print('part of series is stationary !')

    for i in range(n_window_size):
        for j in range(n_step):
            if step_slide[j] > window_size[i] or window_size[i] < window_min:
                print('->Window size {} < moving step {} or min size{}，this running skip'.format(window_size[i], step_slide[j], window_min))
                nk += 1
                continue
            # ====== conventional sliding window method (tail parameter > 3)
            time2 = timeit.default_timer()
            bresult, n_inbound, n_slide = variation_bound(data2_yx, \
              signif_thres, time_lag, window_size[i], step_slide[j], 5)
            time_slide[j, i] += timeit.default_timer() - time2
            bresult, bpack = counts2label(bresult, nsample, clabels)
            acyb = accuracy_rate(bresult, clabels)
            acy_slide[j, i] += acyb

            # ====== different-region balance method ======
            time2 = timeit.default_timer()
            dresult, n_inbound, n_slide = variation_bound(data2_yx, \
              signif_thres, time_lag, window_size[i], step_slide[j])
            time_balance[j, i] += timeit.default_timer() - time2
            inbound_rate[j, i] += n_inbound / n_slide
            dresult, dpack = counts2label(dresult, nsample, clabels)
            acyd = accuracy_rate(dresult, clabels)
            acy_balance[j, i] += acyd
            nk += 1
            time2 = (timeit.default_timer() - time1) / 60
            print('Total running number: {}，this running: {}，time: {:.2f}min，i={}, j={}'.format(n_total, nk, time2, i, j))

time3 = timeit.default_timer() - time1
print('Repetition times: {}，total time: {:.2f}sec.'.format(n_iteration, time3))
print('====== conventional sliding window method ======')
printresult(acy_slide, time_slide, n_iteration, window_size, step_slide)
print('====== different-region balance method ======')
printresult(acy_balance, time_balance, n_iteration, window_size, step_slide)

print('stop')
