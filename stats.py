import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def auto_correlation(x,lag = 1):
    a = pd.Series(np.reshape(x,(-1)))
    b = a.autocorr(lag = lag)
    if np.isnan(b) or np.isinf(b):
        return 0
    return b


def acf(x,max_lag=100):
    acf_result = []
    for i in range(max_lag):
        acf_result.append(auto_correlation(x, lag=i + 1))
    return np.array(acf_result)


def leverage_effect(x ,multiple=False, min_lag=1, max_lag=100):
    def compute_levs(x,x_abs):
        Z = (np.mean(x_abs**2))**2
        second_term = np.mean(x)*np.mean(x_abs**2)
        def compute_for_t(t):
            if t == 0:
                first_term = np.mean(x*(x_abs)**2)
            elif t > 0:
                first_term = np.mean(x[:-t]*(x_abs[t:]**2))
            else:
                first_term = np.mean(x[-t:]*(x_abs[:t]**2) )
            return (first_term-second_term)/Z
        levs = [compute_for_t(t) for t in range(min_lag,max_lag)]
        return np.array(levs)

    x_abs = np.abs(x)
    if multiple:
        levs = np.zeros(max_lag-min_lag)
        for e1,e2 in zip(x,x_abs):
            levs += compute_levs(e1,e2)
        levs /= x.shape[0]
    else:
        levs = compute_levs(x,x_abs)

    return levs

def plot_leverage_effect(x,y,file_name):
    plt.figure(figsize=(16, 9))
    plt.plot(x,y)
    plt.xlabel(r'Lag k',fontsize='xx-large')
    plt.ylabel(r'$L(k)$',fontsize='xx-large')
    plt.title('Leverage Effect', fontsize='xx-large')
    plt.axhline()
    plt.savefig(file_name,transparent=False)
    plt.close()