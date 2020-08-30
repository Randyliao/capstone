import numpy as np
from .seair import SeairOde


def fct_overwhelm(z, I_max, number_points_per_day):
    """
    Computes by how much hospital capacity is overhelmed
    If number of people requiring ICU beds is twice the maximal number Imax for 30 days, returns 1

    :param z:
    :param T:
    :param I_max:
    :return:
    """
    fct_I_max = I_max*np.ones(z.size)
    integral = (1/number_points_per_day)*sum((z-fct_I_max)*(z-fct_I_max>0))
    return (1/30)*(1/I_max)*integral


def overwhelm_vs_release(I_max, cdf_p, cdf_n, T, delta_r, delta_c, grid_prop, p=0.995, N_0=67*10**6, R_0=4*10**6, Inf_0 = 20*10**4, ICU_0=4*10**3, number_points_per_day = 500):
    """

    :param I_max:
    :param cdf_p:
    :param cdf_n:
    :param T:
    :param delta_r:
    :param delta_c:
    :param grid_prop:
    :param p:
    :param N_0:
    :param R_0:
    :param Inf_0:
    :param ICU_0:
    :param number_points_per_day:
    :return:
    """
    overwhelm = []
    for grid_prop_i in grid_prop:
        q = compute_q(p, cdf_p, cdf_n, np.array([grid_prop_i]))
        seair = SeairOde(p=p, N_0=N_0, R_0=R_0, Inf_0=Inf_0, ICU_0=ICU_0, number_points_per_day=number_points_per_day) # TODO : study **kwargs
        [t, sol] = seair.progressive_release(T, q, delta_r, delta_c)
        overwhelm_temp = fct_overwhelm(sol[:, SeairOde.get_y_order().index("ICU_c")] +
                                       sol[:, SeairOde.get_y_order().index("ICU_r")],
                                       I_max,
                                       number_points_per_day)
        overwhelm.append(overwhelm_temp)
    return overwhelm



def generate_time_prop_fixed_frequency(n_days, release_rate):
    """For fixed frequency strategy, generate time and released population proportion

        Keyword arguments:
        n_days -- number of days between releases
        release_rate -- Percentage of released people every n_days
    """
    n = int(np.ceil(100 / release_rate))
    T = np.arange(n_days, n_days*(n+1), n_days)
    prop = np.linspace(release_rate / 100, n * release_rate / 100, n)
    prop[-1] = 1.0
    return T, prop


def dichotomy(f):
    g, d = 0., 1.
    eps = 10**(-5)
    while d-g > eps:
        m = (g+d)/2
        if f(m) == 0:
            return m
        elif f(g) * f(m) < 0:
            d=m
        else:
            g=m
    return (g+d)/2




def invert(p, cdf_p, cdf_n, prop):
    """
    Given p and betas parameters, finds for each period the threshold 'a' which makes sure that the wanted proportion of people is released, i.e., that solves p*cdf_negative(a)+(1-p)*cdf_positive(a) = prop[i]
    In other words, finds the threshold 'a' so that the released population = prop[i]

    :param p: Proportion of people with mild symptoms
    :param cdf_p: cdf function for people with severe symptoms
    :param cdf_n: cdf function for people with mild symptoms
    :param prop: a vector of the population proportion released at each period
    :return: a vector of the same size as prop that solves the above equation
    """
    a = []
    for prop_i in prop:
        fct_temp = lambda t: p*cdf_n(t)+(1-p)*cdf_p(t)-prop_i
        a.append(dichotomy(fct_temp))
    return np.array(a)


def compute_q(p, cdf_p, cdf_n, prop):
    """
    Compute false positive and false negative rates depending on parameters of the beta distribution and chosen proportions

    :param p: Proportion of people with mild symptoms
    :param cdf_p: cdf function for people with severe symptoms
    :param cdf_n: cdf function for people with mild symptoms
    :param prop: a vector of the population proportion released at each period
    :return: a vector of the same size as prop that solves the above equation
    :return: false positive and false negative rates for the chosen proportion of reelased people
    """
    a = invert(p, cdf_p, cdf_n, prop)
    q = []
    for a_i in a:
        # q_FP : people non released but who will have mild symptoms
        q_FP = 1-cdf_n(a_i)
        # q_FN : people released but who will have severe symptoms
        q_FN = cdf_p(a_i)
        q.append(np.array([q_FP, q_FN]))
    return q
