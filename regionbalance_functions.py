# -*- coding: utf-8 -*-
# Created by Kaijun WANG in June - August 2020, adjusted in May 2024

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def syntheticdata_generate(std_a=0.1, tlag=2, nsample=1000, s=450, t=650):
    # generate synthetic 2 time series，casual period [s,t]，delay tlag
    # std_a is used instead of var_b: std_a = math.sqrt(var_b)
    bk = np.random.randint(0, 9, 2, dtype='l')
    # low 0 is inclusive, high 10 is exclusive
    irand = np.random.randint(0, 10, 9, dtype='l') - 4
    s = s - 1 + irand[bk[0]]
    t = t + irand[bk[1]]

    num_t = np.array([i for i in range(nsample)])
    causal_label = np.zeros(nsample, dtype=int)  # 0: no causality
    ak = np.array([0.2 * pow(-1, i+1) for i in range(tlag)])
    bk = np.array([0.2] * tlag)

    # time series Y
    da_y = np.random.uniform(0, 0.1, nsample)
    # time series X，variation by sin，Gaussian noise
    da_x = 1 + np.sin(0.08 * num_t) + np.random.normal(0, std_a, nsample)

    if tlag <= 0:
        tlag2 = 2
    else:
        tlag2 = tlag + tlag
    # part1: no relation [0,s-1]
    for i in range(tlag, t+tlag2):  # +lag: it continues for points around t
        y_lag = da_y[i-tlag: i]
        da_y[i] = np.dot(ak, y_lag) + np.random.normal(0, std_a)
    # smooth several starting points
    da_y[0:tlag2] = da_y[tlag2+1:tlag2+tlag2+1]

    # part2: causal part [s,t-1]
    causal_label[s:t+1] = 1
    for i in range(s, t):
        y_lag = da_y[i-tlag: i]
        x_lag = da_x[i-tlag: i]
        da_y[i] = np.dot(ak, y_lag) + np.dot(bk, x_lag) + np.random.normal(0, std_a)

    # part3: no relation [t,nsample-1]
    # [t:t+lag] continuation from part1, exclude influence of part2 points
    for i in range(t+tlag, nsample):
        y_lag = da_y[i - tlag:i]
        da_y[i] = np.dot(ak, y_lag) + np.random.normal(0, std_a)

    da_x = np.array([da_x], dtype=np.float32)
    da_y = np.array([da_y], dtype=np.float32)
    data_yx = np.hstack((da_y.T, da_x.T))
    print('-->synthetic time series：length {}，noise std {}，causal period [{},{}]；'.format(nsample, std_a, s, t))
    # first column of data_yx is the response/target variable
    da_head = ['feature', 'target']
    return data_yx, causal_label, da_head


def dataload_taxitrips(sw=0):
    data = pd.read_csv('data_taxitrips_a.txt', sep='\t')
    data = np.array(data, dtype=np.float32)
    nsize = data.shape
    causal_label = data[:, 4]
    if sw:  # Pickup-Sweet
        data_yx = data[:, 2:4]
    else:  # Dropoff-Sweet
        da_x = np.array([data[:, 1]], dtype=np.float32)
        da_y = np.array([data[:, 3]], dtype=np.float32)
        data_yx = np.hstack((da_y.T, da_x.T))
    for i in range(0, nsize[0]):
        data_yx[i, 0] = math.sqrt(data_yx[i, 0])
        data_yx[i, 1] = math.sqrt(data_yx[i, 1])
    timelag = 5
    window = [12, 16, 18, 20, 24, 48, 72]
    window_max = 72
    step = [4, 8, 12]
    return data_yx, causal_label, timelag, window, step, window_max


def dataload_FishBaboon(ch=0):
    if ch:
        fname = 'data_BaboonTrajectoryDir1.txt'
        window = [90, 100, 110, 120, 130, 140]  #
        step = [10, 20, 30, 40]  #
        timelag = 12
        widey = 25
        widex = 25
    else:
        fname = 'data_FishTrajectoryDir2.txt'
        window = [130, 140, 150, 160, 170]  #
        step = [10, 20, 30, 40]  #
        timelag = 65
        widey = 10
        widex = 20
    data = pd.read_table(fname, sep='\t')  #, header=None
    data = np.array(data, dtype=np.float32)
    da = pd.Series(data[:, 0])
    da_y = da.rolling(window=widey).mean()
    da = pd.Series(data[:, 1])
    da_x = da.rolling(window=widex).mean()
    for i in range(widey - 1):
        da_y[i] = da_y[widey-1]
    for i in range(widex-1):
        da_x[i] = da_x[widex-1]
    da_x = np.array([da_x], dtype=np.float32)
    da_y = np.array([da_y], dtype=np.float32)
    data_yx = np.hstack((da_y.T, da_x.T))
    if ch:
        i
    else:
        data_yx = p_sqrt(data_yx)
    causal_label = data[:, 2]
    window_max = len(causal_label)
    # plotdata(data_yx,0,0)
    return data_yx, causal_label, timelag, window, step, window_max


def p_sqrt(data):
    ns = data.shape
    for j in range(0, ns[1]):
        for i in range(0, ns[0]):
            sg = np.sign(data[i, j])
            data[i, j] = sg * math.sqrt(sg * data[i, j])
    return data


def counts2label(result, nsize, labels=[0, 0]):
    qc = np.zeros((nsize, 1), dtype=int)  # count time points where causality occurs
    qn = np.zeros((nsize, 1), dtype=int)  # count time points where causality disappears
    labels = np.array([labels], dtype=int)
    ns = labels.shape
    qout = np.zeros(nsize, dtype=int)
    for item in result:
        i, j = item[0], item[1]
        val = result[item]
        if val == 0:
            for k in range(i, j + 1):
                qn[k] += 1
        else:
            for k in range(i, j + 1):
                qc[k] += 1
    for i in range(0, nsize):
        if qc[i] > qn[i]:
                qout[i] = 1  # 1: causality appears
    if ns[1] > 5:
        q_all = np.hstack((qc, qn))
        q_all = np.hstack((q_all, labels.T))
        return qout, q_all
    return qout


def counts2score(result, nsize, labels=[0, 0]):
    qc = np.zeros((nsize, 1), dtype=np.float)
    qn = np.zeros((nsize, 1), dtype=np.float)
    score = np.zeros((nsize, 1), dtype=np.float)
    qout = np.zeros(nsize, dtype=int)
    labels = np.array([labels], dtype=int)
    ns = labels.shape
    smax = 0.0
    for item in result:
        i, j = item[0], item[1]
        val = result[item]
        if val == 0:
            for k in range(i, j + 1):
                qn[k] += 1
        else:
            for k in range(i, j + 1):
                qc[k] += 1
    for i in range(0, nsize):
        val = int(qc[i] + qn[i])
        if val > 0:
            val = int(qc[i])/val
            score[i] = val  # causal score
            if val > smax:
                smax = val
    smax = 0.9 * smax
    for i in range(0, nsize):
        if score[i] > smax:
            qout[i] = 1
    if ns[1] > 5:
        q_all = np.hstack((qc, qn))
        q_all = np.hstack((q_all, score))
        return qout, q_all
    return qout


def accuracy_rate(result, causal_label):
    nsize = len(causal_label)
    acy = 0
    for i in range(0, nsize):
        if result[i] == causal_label[i]:
            acy += 1
    acy /= nsize
    return acy


def printresult(acys, atime, nloop, wide, step_slide, rin=0, vb=0):
    # computation of average accuracy and running time
    ns = acys.shape
    for i in range(ns[1]):
        for j in range(ns[0]):
            acy = acys[j, i]/nloop
            tim = atime[j, i]/nloop
            print('->Window size:{}, step size:{},'.format(wide[i], step_slide[j]), end=' ')
            print('Acy_average{:.2%}, time:{:.2f}sec.'.format(acy, tim))
            if vb:
                print('small region number/total window number{:.2%};'.format(rin[j, i]))
            elif rin:
                return

def plotdata(data, s, t):
    num_t = range(1, data.shape[0]+1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    plt.plot(num_t, data[:, 1], color='black', linewidth=0.9, label='X(cause)')
    plt.plot(num_t, data[:, 0], color='blue', linewidth=0.9, label='Y(target)')
    plt.title('causal region scope [{},{}]'.format(s, t))
    plt.show()
