# -*- coding: utf-8 -*-
# Created by Kaijun WANG in August 2020, adjusted in May 2024

import numpy as np
from regionbalance_inference import variation_bound
from regionbalance_functions import syntheticdata_generate, counts2label, accuracy_rate


a_datset = 1
window_size = [20, 40, 80]
step = 5
ftest_thred = 0.05
if a_datset == 1:
    n_repeat = 2
else:
    n_repeat = 1
acy_matrix = np.zeros([2, n_repeat], float)

for k in range(n_repeat):
    if a_datset == 1:
        nsample = 500  # length of time series
        lag_max = 2
        dat_yx, labels, temp = syntheticdata_generate(0.20, lag_max, nsample, s=150, t=350)  # default 450,650

    # the different-region balance method, conventional sliding window method
    acy_region_balance = 0
    acy_slide_window = 0
    for wsize in window_size:
        bresult, ntemp1, ntemp2 = variation_bound(dat_yx[0:nsample, :], ftest_thred, lag_max, wsize, step)
        result, lab_pack = counts2label(bresult, nsample, labels)
        acy_temp = accuracy_rate(result, labels)
        acy_region_balance += acy_temp

        # conventional sliding window method (tail parameter > 3)
        sresult, ntemp1, ntemp2 = variation_bound(dat_yx[0:nsample, :], ftest_thred, lag_max, wsize, step, 5)  # 5: sliding window
        result, lab_pack = counts2label(sresult, nsample, labels)
        acy_temp = accuracy_rate(result, labels)
        acy_slide_window += acy_temp

    acy_matrix[0, k] = acy_region_balance / 3
    acy_matrix[1, k] = acy_slide_window / 3
    np.savetxt('zaccuracy.txt', acy_matrix, fmt="%.4f", delimiter="\t")

'''
    # prepare the dataset by variable position exchange
    dat_xy = dat_yx[0:nsample, :].copy()
    dat_xy[:, 0] = dat_yx[0:nsample, 1]
    dat_xy[:, 1] = dat_yx[0:nsample, 0]
'''
