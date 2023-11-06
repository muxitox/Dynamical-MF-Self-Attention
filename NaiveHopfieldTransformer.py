import numpy as np
import matplotlib.pyplot as plt
import os
class NaiveHopfieldTransformer:

    def __init__(self, beta, m0, Wodd, Weven):
        self.beta = beta
        self.m0 = m0
        self.Wodd = Wodd
        self.Weven = Weven


    def simulate(self, max_steps):

        old_m = self.m0
        m_odd = []
        m_even = []
        deriv_beta_odd = []
        deriv_beta_even = []


        for i in range(0, max_steps):

            if i % 2 == 0:
                m = np.tanh(self.beta * self.Wodd * old_m)
                deriv_beta = self.Wodd * old_m * (1 - m**2)
                m_odd.append(m)
                deriv_beta_odd.append(deriv_beta)

            else:
                m = np.tanh(self.beta * self.Weven * old_m)
                deriv_beta = self.Weven * old_m * (1 - m**2)
                m_even.append(m)
                deriv_beta_even.append(deriv_beta)


            old_m = m

        return m_odd, m_even, deriv_beta_odd, deriv_beta_even


def plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven):

    plt.figure()
    plt.plot(range(0, len(m_odd)), m_odd, label=f'm_odd')
    plt.plot(range(0, len(m_even)), m_even, label='m_even')
    plt.xlabel('t')
    plt.title(f'W_odd={Wodd} W_even={Weven} Beta={beta}, m0={m0}')
    plt.legend()
    # plt.show()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/m0_{m0}_beta_{beta}.png')
    plt.close()

def plot_save_phase(beta_list, last_m_odd_list, last_m_even_list, m0_list, Wodd, Weven):

    for i in range(len(m0_list)):

        last_m_odd_list_m0 = [l[i] for l in last_m_odd_list]
        last_m_even_list_m0 = [l[i] for l in last_m_even_list]

        plt.figure()
        plt.plot(beta_list, last_m_odd_list_m0, label='m_odd')
        plt.plot(beta_list, last_m_even_list_m0, label='m_even')
        plt.xlabel('beta')
        plt.title(f'W_odd={Wodd} W_even={Weven} m0={m0_list[i]}')
        plt.legend()
        plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/m0_{m0_list[i]}_phase.png')
        # plt.show()
        plt.close()

def plot_save_bifurcation(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven):
    plt.figure()
    plt.plot(beta_list_shallow, last_m_odd_list_shallow, label='m_odd', ls='', marker='o')
    plt.plot(beta_list_shallow, last_m_even_list_shallow, label='m_even', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/bifurcation.png')
    plt.show()
    plt.close()

def plot_save_bifurcation_angle(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven):

    angle = np.rad2deg(np.arctan2(last_m_odd_list_shallow, last_m_even_list_shallow))
    abs_val = np.sqrt(np.square(last_m_odd_list_shallow) + np.square(last_m_even_list_shallow))
    plt.figure()
    plt.plot(beta_list_shallow, angle, label='angle', ls='', marker='o')
    plt.plot(beta_list_shallow, abs_val, label='abs', ls='', marker='o')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/bifurcation_polar.png')
    plt.show()
    plt.close()


def plot_phase(Wodd, Weven, last_deriv_beta_odd_shallow, last_deriv_beta_even_shallow, beta_list_shallow):
    plt.figure()
    plt.plot(beta_list_shallow, last_deriv_beta_odd_shallow, label='grad_even', ls='', marker='o')
    plt.plot(beta_list_shallow, last_deriv_beta_even_shallow, label='grad_odd', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/phase_plane.png')
    plt.show()
    plt.close()


def plot_phase2(Wodd, Weven, last_m_odd_list_shallow, last_m_even_list_shallow, beta_list_shallow):

    last_deriv_beta_odd = np.gradient(last_m_odd_list_shallow, beta_list_shallow)
    last_deriv_beta_even = np.gradient(last_m_even_list_shallow, beta_list_shallow)

    plt.figure()
    plt.plot(beta_list_shallow, last_deriv_beta_odd, label='grad_even', ls='', marker='o')
    plt.plot(beta_list_shallow, last_deriv_beta_even, label='grad_odd', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}/phase_plane2.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    Wodd = 0.1
    Weven = 0.6

    if not os.path.exists(f"imgs/Wodd_{Wodd}_Weven_{Weven}/"):
        os.makedirs(f"imgs/Wodd_{Wodd}_Weven_{Weven}/")

    beta_list = np.linspace(0,30, 150)

    # m0_list = [-1, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 1  ]
    m0_list = [0.1, 0.3, 0.5, 0.7, 1]

    last_m_odd_list = []
    last_m_even_list = []

    last_m_odd_list_shallow = []
    last_m_even_list_shallow = []

    last_deriv_beta_odd = []
    last_deriv_beta_even = []
    last_deriv_beta_odd_shallow = []
    last_deriv_beta_even_shallow = []

    beta_list_shallow = []

    for beta in beta_list:

        last_m_odd_list_m0 = []
        last_m_even_list_m0 = []

        for m0 in m0_list:

            NHT = NaiveHopfieldTransformer(beta, m0, Wodd, Weven)

            max_steps = 500
            m_odd, m_even, deriv_beta_odd, deriv_beta_even = NHT.simulate(max_steps)

            last_m_odd_list_m0.append(m_odd[-1])
            last_m_even_list_m0.append(m_even[-1])

            last_deriv_beta_odd.append(deriv_beta_odd[-1])
            last_deriv_beta_even.append(deriv_beta_even[-1])

            last_m_odd_list_shallow.extend(m_odd[-2:])
            last_m_even_list_shallow.extend(m_even[-2:])
            last_deriv_beta_odd_shallow.extend(deriv_beta_odd[-2:])
            last_deriv_beta_even_shallow.extend(deriv_beta_even[-2:])
            beta_list_shallow.extend([beta, beta])

            # plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven)

        last_m_odd_list.append(last_m_odd_list_m0)
        last_m_even_list.append(last_m_even_list_m0)

    # plot_save_phase(beta_list, last_m_odd_list, last_m_even_list, m0_list, Wodd, Weven)

    plot_save_bifurcation(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven)
    plot_save_bifurcation_angle(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven)

    plot_phase(Wodd, Weven, last_deriv_beta_odd_shallow, last_deriv_beta_even_shallow, beta_list_shallow)

    ini_cond_idx = 0
    num_ini_conds = len(m0_list)
    plot_phase2(Wodd, Weven, last_m_odd_list_shallow[ini_cond_idx::num_ini_conds], last_m_even_list_shallow[ini_cond_idx::num_ini_conds], beta_list_shallow[ini_cond_idx::num_ini_conds])
