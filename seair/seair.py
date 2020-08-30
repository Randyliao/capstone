import numpy as np
from scipy.integrate import odeint


class SeairOde:

    def __init__(self, p=0.995, N_0=67*10**6, R_0=4*10**6, Inf_0 = 20*10**4, ICU_0=4*10**3, number_points_per_day = 500):
        ######################
        ## Time discretisation
        ######################
        self.number_points_per_day = number_points_per_day

        ######################
        # Parameters
        ######################
        self.parameters = {
            'p': p,             # Proportion of people with mild symptoms
            'Reprod_0': 3.3,    # basic reproduction number
            'eps': 1 / 3.7,     # Rate of passage from E to A
            'sigma': 1 / 1.5,    # Rate of passage from A to I
            'gamma_m': 1 / 2.3,  # Recovery rate for mild symptoms
            'gamma_s': 1 / 17.,  # Recovery rate for severe symptoms
            'alpha': 1 / 11.7,
            'eta': 1 / 7.,


        }

        ######################
        # Initial conditions
        ######################
        var = 1/(1+self.parameters['eps']*((1/self.parameters['sigma'])+(1/self.parameters['gamma_m'])))
        E_0 = var*Inf_0
        A_0 = (self.parameters['eps']/self.parameters['sigma'])*E_0
        I_0 = (self.parameters['eps']/self.parameters['gamma_m'])*E_0
        S_0 = N_0 - (E_0 + A_0 + I_0 + ICU_0 + R_0)
        self.initial_conditions = {'N_0': N_0,  # Total number of individuals at time 0 # Todo : comments okay ?
                                   'R_0': R_0,  # Total number of immune at time 0
                                   'Inf_0': Inf_0, # Total number of infected E + A + I at time 0
                                   'I_0': I_0,  # Total number of asymptomatic  at time 0
                                   'S_0': S_0,  # Total number of susceptibles at time 0
                                   'E_0': E_0,  # Total number of infected at time 0
                                   'A_0': A_0,   # Total number of latent at time 0
                                   'ICU_0': ICU_0
                                   }

        # Finalize parameters
        self.parameters['beta'] = (self.parameters['Reprod_0'] / self.initial_conditions['N_0']) / \
                                  (self.parameters['p']*(1/self.parameters['gamma_m'] + 1/self.parameters['sigma']) +
                                   (1-self.parameters['p'])*(1/self.parameters['eta'] + 1/self.parameters['sigma']) )


        ######################
        # Computing initial conditions in each category as if labeling were perfect
        ######################
        self.initial_conditions_category = {
            # S
            'S_m_0': self.parameters['p'] * self.initial_conditions['S_0'],
            'S_s_0': (1 - self.parameters['p']) * self.initial_conditions['S_0'],
            'S_mr_0':self.parameters['p'] * self.initial_conditions['S_0'],
            'S_mc_0': 0,
            'S_sc_0': (1 - self.parameters['p']) * self.initial_conditions['S_0'],
            'S_sr_0': 0,
            # A
            'A_m_0': self.parameters['p'] * self.initial_conditions['A_0'],
            'A_s_0': (1 - self.parameters['p']) * self.initial_conditions['A_0'],
            'A_mr_0': self.parameters['p'] * self.initial_conditions['A_0'],
            'A_mc_0': 0,
            'A_sc_0': (1 - self.parameters['p']) * self.initial_conditions['A_0'],
            'A_sr_0': 0,
            # E
            'E_m_0': self.parameters['p'] * self.initial_conditions['E_0'],
            'E_s_0': (1 - self.parameters['p']) * self.initial_conditions['E_0'],
            'E_mr_0': self.parameters['p'] * self.initial_conditions['E_0'],
            'E_mc_0': 0,
            'E_sc_0': (1 - self.parameters['p']) * self.initial_conditions['E_0'],
            'E_sr_0': 0,
            # I
            'I_m_0': self.parameters['p'] * self.initial_conditions['I_0'],
            'I_s_0': (1 - self.parameters['p']) * self.initial_conditions['I_0'],
            'I_mr_0': self.parameters['p'] * self.initial_conditions['I_0'],
            'I_mc_0': 0,
            'I_sc_0': (1 - self.parameters['p']) * self.initial_conditions['I_0'],
            'I_sr_0': 0,
            # ICU
            'ICU_c_0': self.initial_conditions['ICU_0'],
            'ICU_r_0': 0,
            # R
            'R_m_0': self.parameters['p'] * self.initial_conditions['R_0'],
            'R_s_0': (1 - self.parameters['p']) * self.initial_conditions['R_0'],
            'R_mr_0': self.parameters['p'] * self.initial_conditions['R_0'],
            'R_mc_0': 0,
            'R_sc_0': (1 - self.parameters['p']) * self.initial_conditions['R_0'],
            'R_sr_0': 0,
            # D
            'D_0' : 0
        }

        ######################
        # Initial condition without any misclassification
        ######################
        self.y_0 = np.array([self.initial_conditions_category['S_mr_0'],
                             self.initial_conditions_category['E_mr_0'],
                             self.initial_conditions_category['A_mr_0'],
                             self.initial_conditions_category['I_mr_0'],
                             self.initial_conditions_category['R_mr_0'],
                             self.initial_conditions_category['S_mc_0'],
                             self.initial_conditions_category['E_mc_0'],
                             self.initial_conditions_category['A_mc_0'],
                             self.initial_conditions_category['I_mc_0'],
                             self.initial_conditions_category['R_mc_0'],
                             self.initial_conditions_category['S_sc_0'],
                             self.initial_conditions_category['E_sc_0'],
                             self.initial_conditions_category['A_sc_0'],
                             self.initial_conditions_category['I_sc_0'],
                             self.initial_conditions_category['ICU_c_0'],
                             self.initial_conditions_category['R_sc_0'],
                             self.initial_conditions_category['S_sr_0'],
                             self.initial_conditions_category['E_sr_0'],
                             self.initial_conditions_category['A_sr_0'],
                             self.initial_conditions_category['I_sr_0'],
                             self.initial_conditions_category['ICU_r_0'],
                             self.initial_conditions_category['R_sr_0'],
                             self.initial_conditions_category['D_0']])


    def progressive_release(self, T, q, delta_r, delta_c):
        """

        :param T:
        :param q:
        :param delta_r:
        :param delta_c:
        :param y_0:
        :return:
        """
        n = T.size
        t_tot = []
        sol_tot = []
        t_temp = np.linspace(0, T[0], T[0] * self.number_points_per_day)
        ## Update initial condition according to q[0]
        real_y_0 = SeairOde.shuffle_qs(q[0], self.y_0)
        sol_temp = self.prop_ODE(delta_r, delta_c, real_y_0, T[0], T[0] * self.number_points_per_day)
        sol_tot = sol_temp
        t_tot = t_temp
        for i in range(1, n):
            y_temp = SeairOde.shuffle_qs(q[i], sol_temp[-1, :])
            sol_temp = self.prop_ODE(delta_r, delta_c, y_temp, T[i] - T[i - 1],
                                (T[i] - T[i - 1]) * self.number_points_per_day)
            sol_tot = np.concatenate([sol_tot[:-1], sol_temp])
            t_temp = np.linspace(T[i - 1], T[i], (T[i] - T[i - 1]) * self.number_points_per_day)
            t_tot = np.append(t_tot[:-1], t_temp)
        return t_tot, sol_tot

    @staticmethod
    def shuffle_qs(new_q, old_y):
        """
        Computes new compartments from old ones when q's are changed

        :param old_y:
        :param new_q:
        :return:
        """
        new_q_FP = new_q[0]
        new_q_FN = new_q[1]
        new_S_mr = (1 - new_q_FP) * (old_y[SeairOde.get_y_order().index("S_mr")] + old_y[SeairOde.get_y_order().index("S_mc")])
        new_E_mr = (1 - new_q_FP) * (old_y[SeairOde.get_y_order().index("E_mr")] + old_y[SeairOde.get_y_order().index("E_mc")])
        new_A_mr = (1 - new_q_FP) * (old_y[SeairOde.get_y_order().index("A_mr")] + old_y[SeairOde.get_y_order().index("A_mc")])
        new_I_mr = (1 - new_q_FP) * (old_y[SeairOde.get_y_order().index("I_mr")] + old_y[SeairOde.get_y_order().index("I_mc")])
        new_R_mr = (1 - new_q_FP) * (old_y[SeairOde.get_y_order().index("R_mr")] + old_y[SeairOde.get_y_order().index("R_mc")])
        new_S_mc = new_q_FP * (old_y[SeairOde.get_y_order().index("S_mr")] + old_y[SeairOde.get_y_order().index("S_mc")])
        new_E_mc = new_q_FP * (old_y[SeairOde.get_y_order().index("E_mr")] + old_y[SeairOde.get_y_order().index("E_mc")])
        new_A_mc = new_q_FP * (old_y[SeairOde.get_y_order().index("A_mr")] + old_y[SeairOde.get_y_order().index("A_mc")])
        new_I_mc = new_q_FP * (old_y[SeairOde.get_y_order().index("I_mr")] + old_y[SeairOde.get_y_order().index("I_mc")])
        new_R_mc = new_q_FP * (old_y[SeairOde.get_y_order().index("R_mr")] + old_y[SeairOde.get_y_order().index("R_mc")])
        new_S_sc = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("S_sc")] + old_y[SeairOde.get_y_order().index("S_sr")])
        new_E_sc = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("E_sc")] + old_y[SeairOde.get_y_order().index("E_sr")])
        new_A_sc = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("A_sc")] + old_y[SeairOde.get_y_order().index("A_sr")])
        new_I_sc = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("I_sc")] + old_y[SeairOde.get_y_order().index("I_sr")])
        new_ICU_c = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("ICU_c")] + old_y[SeairOde.get_y_order().index("ICU_r")])
        new_R_sc = (1 - new_q_FN) * (old_y[SeairOde.get_y_order().index("R_sc")] + old_y[SeairOde.get_y_order().index("R_sr")])
        new_S_sr = new_q_FN * (old_y[SeairOde.get_y_order().index("S_sc")] + old_y[SeairOde.get_y_order().index("S_sr")])
        new_E_sr = new_q_FN * (old_y[SeairOde.get_y_order().index("E_sc")] + old_y[SeairOde.get_y_order().index("E_sr")])
        new_A_sr = new_q_FN * (old_y[SeairOde.get_y_order().index("A_sc")] + old_y[SeairOde.get_y_order().index("A_sr")])
        new_I_sr = new_q_FN * (old_y[SeairOde.get_y_order().index("I_sc")] + old_y[SeairOde.get_y_order().index("I_sr")])
        new_ICU_r = new_q_FN * (old_y[SeairOde.get_y_order().index("ICU_c")] + old_y[SeairOde.get_y_order().index("ICU_r")])
        new_R_sr = new_q_FN * (old_y[SeairOde.get_y_order().index("R_sc")] + old_y[SeairOde.get_y_order().index("R_sr")])
        new_D = old_y[SeairOde.get_y_order().index("D")]
        return np.array([
            new_S_mr,
            new_E_mr,
            new_A_mr,
            new_I_mr,
            new_R_mr,
            new_S_mc,
            new_E_mc,
            new_A_mc,
            new_I_mc,
            new_R_mc,
            new_S_sc,
            new_E_sc,
            new_A_sc,
            new_I_sc,
            new_ICU_c,
            new_R_sc,
            new_S_sr,
            new_E_sr,
            new_A_sr,
            new_I_sr,
            new_ICU_r,
            new_R_sr,
            new_D])

    @staticmethod
    def get_y_order():
        return ['S_mr',
                'E_mr',
                'A_mr',
                'I_mr',
                'R_mr',
                'S_mc',
                'E_mc',
                'A_mc',
                'I_mc',
                'R_mc',
                'S_sc',
                'E_sc',
                'A_sc',
                'I_sc',
                'ICU_c',
                'R_sc',
                'S_sr',
                'E_sr',
                'A_sr',
                'I_sr',
                'ICU_r',
                'R_sr',
                'D']

    def ODE_function(self, delta_r, delta_c):
        def lda_tot(y, beta):
            return beta * (effective_A(y, delta_r, delta_c) + effective_I(y, delta_r, delta_c))

        def lda_m(y, beta):
            return (1 - delta_r) * lda_tot(y, beta)

        def lda_s(y, beta):
            return (1 - delta_c) * lda_tot(y, beta)

        # For S variables
        def f_S_mr(y, beta):
            return -lda_m(y, beta) * y[SeairOde.get_y_order().index("S_mr")]

        def f_S_mc(y, beta):
            return -lda_s(y, beta) * y[SeairOde.get_y_order().index("S_mc")]

        def f_S_sc(y, beta):
            return -lda_s(y, beta) * y[SeairOde.get_y_order().index("S_sc")]

        def f_S_sr(y, beta):
            return -lda_m(y, beta) * y[SeairOde.get_y_order().index("S_sr")]

        # For E variables
        def f_E_mr(y, beta, eps):
            return lda_m(y, beta) * y[SeairOde.get_y_order().index("S_mr")] - eps * y[SeairOde.get_y_order().index("E_mr")]

        def f_E_mc(y, beta, eps):
            return lda_s(y, beta) * y[SeairOde.get_y_order().index("S_mc")] - eps * y[SeairOde.get_y_order().index("E_mc")]

        def f_E_sc(y, beta, eps):
            return lda_s(y, beta) * y[SeairOde.get_y_order().index("S_sc")] - eps * y[SeairOde.get_y_order().index("E_sc")]

        def f_E_sr(y, beta, eps):
            return lda_m(y, beta) * y[SeairOde.get_y_order().index("S_sr")] - eps * y[SeairOde.get_y_order().index("E_sr")]

        # For A variables
        def f_A_mr(y, eps, sigma):
            return eps * y[SeairOde.get_y_order().index("E_mr")] - sigma * y[SeairOde.get_y_order().index("A_mr")]

        def f_A_mc(y, eps, sigma):
            return eps * y[SeairOde.get_y_order().index("E_mc")] - sigma * y[SeairOde.get_y_order().index("A_mc")]

        def f_A_sc(y, eps, sigma):
            return eps * y[SeairOde.get_y_order().index("E_sc")] - sigma * y[SeairOde.get_y_order().index("A_sc")]

        def f_A_sr(y, eps, sigma):
            return eps * y[SeairOde.get_y_order().index("E_sr")] - sigma * y[SeairOde.get_y_order().index("A_sr")]

        # For I variables
        def f_I_mr(y, sigma, gamma_m):
            return sigma * y[SeairOde.get_y_order().index("A_mr")] - gamma_m * y[SeairOde.get_y_order().index("I_mr")]

        def f_I_mc(y, sigma, gamma_m):
            return sigma * y[SeairOde.get_y_order().index("A_mc")] - gamma_m * y[SeairOde.get_y_order().index("I_mc")]

        def f_I_sc(y, sigma, eta):
            return sigma * y[SeairOde.get_y_order().index("A_sc")] - eta * y[SeairOde.get_y_order().index("I_sc")]

        def f_I_sr(y, sigma, eta):
            return sigma * y[SeairOde.get_y_order().index("A_sr")] - eta * y[SeairOde.get_y_order().index("I_sr")]

        # For ICU variables
        def f_ICU_c(y, eta, gamma_s, alpha):
            return eta * y[SeairOde.get_y_order().index("I_sc")] - (gamma_s + alpha) * y[SeairOde.get_y_order().index("ICU_c")]

        def f_ICU_r(y, eta, gamma_s, alpha):
            return eta * y[SeairOde.get_y_order().index("I_sr")] - (gamma_s + alpha) * y[SeairOde.get_y_order().index("ICU_r")]

        # For R variables
        def f_R_mr(y, gamma_m):
            return gamma_m * y[SeairOde.get_y_order().index("I_mr")]

        def f_R_mc(y, gamma_m):
            return gamma_m * y[SeairOde.get_y_order().index("I_mc")]

        def f_R_sc(y, gamma_s):
            return gamma_s * y[SeairOde.get_y_order().index("ICU_c")]

        def f_R_sr(y, gamma_s):
            return gamma_s * y[SeairOde.get_y_order().index("ICU_r")]

        # For the D variable
        def f_D(y, alpha):
            return alpha*ICU(y)

        # Full function
        return lambda t, y: [f_S_mr(y, self.parameters['beta']),
                             f_E_mr(y, self.parameters['beta'], self.parameters['eps']),
                             f_A_mr(y, self.parameters['eps'], self.parameters['sigma']),
                             f_I_mr(y, self.parameters['sigma'], self.parameters['gamma_m']),
                             f_R_mr(y, self.parameters['gamma_m']),
                             f_S_mc(y, self.parameters['beta']),
                             f_E_mc(y, self.parameters['beta'], self.parameters['eps']),
                             f_A_mc(y, self.parameters['eps'], self.parameters['sigma']),
                             f_I_mc(y, self.parameters['sigma'], self.parameters['gamma_m']),
                             f_R_mc(y, self.parameters['gamma_m']),
                             f_S_sc(y, self.parameters['beta']),
                             f_E_sc(y, self.parameters['beta'], self.parameters['eps']),
                             f_A_sc(y, self.parameters['eps'], self.parameters['sigma']),
                             f_I_sc(y, self.parameters['sigma'], self.parameters['eta']),
                             f_ICU_c(y, self.parameters['eta'], self.parameters['gamma_s'], self.parameters['alpha']),
                             f_R_sc(y, self.parameters['gamma_s']),
                             f_S_sr(y, self.parameters['beta']),
                             f_E_sr(y, self.parameters['beta'], self.parameters['eps']),
                             f_A_sr(y, self.parameters['eps'], self.parameters['sigma']),
                             f_I_sr(y, self.parameters['sigma'], self.parameters['eta']),
                             f_ICU_r(y, self.parameters['eta'], self.parameters['gamma_s'], self.parameters['alpha']),
                             f_R_sr(y, self.parameters['gamma_s']),
                             f_D(y, self.parameters['alpha'])]

    def prop_ODE(self, delta_r, delta_c, y_init, final_time, ndiscr):
        """
        Solves the ODE and returns solution over time

        :param delta_r:
        :param delta_c:
        :param y_init:
        :param final_time:
        :param ndiscr:
        :return:
        """
        t = np.linspace(0, final_time, ndiscr)
        sol = odeint(self.ODE_function(delta_r, delta_c), y_init, t, tfirst=True)
        return sol


    def get_N_0(self):
        return self.initial_conditions['N_0']


######################
## S
######################
def S_m(y):
    return y[SeairOde.get_y_order().index("S_mr")]+y[SeairOde.get_y_order().index("S_mc")]


def S_s(y):
    return y[SeairOde.get_y_order().index("S_sc")]+y[SeairOde.get_y_order().index("S_sr")]


def S(y):
    return S_m(y) + S_s(y)


######################
## E
######################
def E_m(y):
    return y[SeairOde.get_y_order().index("E_mr")]+y[SeairOde.get_y_order().index("E_mc")]


def E_s(y):
    return y[SeairOde.get_y_order().index("E_sc")]+y[SeairOde.get_y_order().index("E_sr")]


def E(y):
    return E_m(y) + E_s(y)


######################
## A
######################
def A_m(y):
    return y[SeairOde.get_y_order().index("A_mr")]+y[SeairOde.get_y_order().index("A_mc")]


def A_s(y):
    return y[SeairOde.get_y_order().index("A_sc")]+y[SeairOde.get_y_order().index("A_sr")]


def A_cm(y):
    return y[SeairOde.get_y_order().index("A_mr")]+y[SeairOde.get_y_order().index("A_sr")]


def A_cs(y):
    return y[SeairOde.get_y_order().index("A_mc")]+y[SeairOde.get_y_order().index("A_sc")]


def A(y):
    return A_m(y) + A_s(y)


######################
## I
######################
def I_m(y):
    return y[SeairOde.get_y_order().index("I_mr")]+y[SeairOde.get_y_order().index("I_mc")]


def I_s(y):
    return y[SeairOde.get_y_order().index("I_sc")]+y[SeairOde.get_y_order().index("I_sr")]


def I_cm(y):
    return y[SeairOde.get_y_order().index("I_mr")]+y[SeairOde.get_y_order().index("I_sr")]


def I_cs(y):
    return y[SeairOde.get_y_order().index("I_mc")]+y[SeairOde.get_y_order().index("I_sc")]


def I(y):
    return I_m(y) + I_s(y)


######################
## ICU
######################
def ICU(y):
    return y[SeairOde.get_y_order().index("ICU_c")] + y[SeairOde.get_y_order().index("ICU_r")]


######################
## R
######################
def R_m(y):
    return y[SeairOde.get_y_order().index("R_mr")]+y[SeairOde.get_y_order().index("R_mc")]


def R_s(y):
    return y[SeairOde.get_y_order().index("R_sc")]+y[SeairOde.get_y_order().index("R_sr")]


def R(y):
    return R_m(y) + R_s(y)


######################
## N
######################
def N(y):
    return S(y)+E(y)+A(y)+I(y)+ICU(y)+R(y)


######################
## Effective contact
######################
def effective_A(y,delta_r,delta_c):
    return (1-delta_r)*A_cm(y) + (1-delta_c)*A_cs(y)


def effective_I(y,delta_r,delta_c):
    return (1-delta_r)*I_cm(y) + (1-delta_c)*I_cs(y)

