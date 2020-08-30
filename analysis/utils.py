import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from sklearn import metrics
from os import path


def plot_densities(alpha_p, beta_p, alpha_n, beta_n, fig_path=None):
    """Plot densities for :
        * proba to have severe symptoms
        * proba to have mild symptoms

        Keyword arguments:
        alpha_p -- alpha for beta law for severe symptoms
        beta_p -- beta for beta law for severe symptoms
        alpha_n -- alpha for beta law for mild symptoms
        beta_n -- beta for beta law for mild symptoms
    """
    prob_grid = np.linspace(0,1,10000)
    positive_pdf = stat.beta.pdf(prob_grid, alpha_p, beta_p)
    negative_pdf = stat.beta.pdf(prob_grid, alpha_n, beta_n)
    plt.plot(prob_grid, positive_pdf, 'b', label = 'Severe symptoms')
    plt.plot(prob_grid, negative_pdf, 'r', label = 'Mild symptoms')
    plt.xlabel('a')
    plt.title('Probability densities for severe and mild symptoms')
    plt.legend()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(path.join(fig_path, 'populations_densities.png'))


def compute_auc(alpha_p, beta_p, alpha_n, beta_n):
    """

    :param alpha_p:
    :param beta_p:
    :param alpha_n:
    :param beta_n:
    :return:
    """
    a = np.linspace(0,1,1000)
    q_FP = []
    q_TP = []
    for i in range(1000):
        q_FP.append(1-stat.beta.cdf(a[i], alpha_n, beta_n))
        q_TP.append(1-stat.beta.cdf(a[i], alpha_p, beta_p))
    return metrics.auc(q_FP, q_TP)
