import numpy as np
import matplotlib.pyplot as plt
import os

imgs_root = "imgs/one_cond"

class NaiveHopfieldTransformer:

    def __init__(self, beta, m0, Wodd, Weven):
        self.beta = beta
        self.m0 = m0
        self.Wodd = Wodd
        self.Weven = Weven


    def simulate(self, max_steps):

        m_even = self.m0
        m_odd = 0
        m_odd_list = []
        m_even_list = []
        deriv_beta_odd = []
        deriv_beta_even = []


        for i in range(1, max_steps):

            if i % 2 == 0:
                m_even = np.tanh(self.beta * self.Weven * m_odd)

                deriv_beta_num = (1 - m_even ** 2) * (
                        self.Weven * m_odd + self.beta * self.Weven * (1 - m_odd ** 2) * self.Wodd * m_even)
                deriv_beta_denom = (
                        1 - (1 - m_even ** 2) * self.beta * self.Weven * (1 - m_odd ** 2) * self.beta * self.Wodd)
                deriv_beta = deriv_beta_num / deriv_beta_denom

                m_even_list.append(m_even)
                deriv_beta_even.append(deriv_beta)

            else:

                m_odd = np.tanh(self.beta * self.Wodd * m_even)

                deriv_beta_num = (1 - m_odd ** 2) * (
                        self.Wodd * m_even + self.beta * self.Wodd * (1 - m_even ** 2) * self.Weven * m_odd)
                deriv_beta_denom = (
                        1 - (1 - m_odd ** 2) * self.beta * self.Wodd * (1 - m_even ** 2) * self.beta * self.Weven)
                deriv_beta = deriv_beta_num / deriv_beta_denom

                m_odd_list.append(m_odd)
                deriv_beta_odd.append(deriv_beta)

        return m_odd_list, m_even_list, deriv_beta_odd, deriv_beta_even


def plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven):

    plt.figure()
    plt.plot(range(0, len(m_odd)), m_odd, label=f'm_odd')
    plt.plot(range(0, len(m_even)), m_even, label='m_even')
    plt.xlabel('t')
    plt.title(f'W_odd={Wodd} W_even={Weven} Beta={beta}, m0={m0}')
    plt.legend()
    # plt.show()
    plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/m0_{m0}_beta_{beta}.png')
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
        plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/m0_{m0_list[i]}_phase.png')
        # plt.show()
        plt.close()

def plot_save_bifurcation(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven):
    plt.figure()
    plt.plot(beta_list_shallow, last_m_odd_list_shallow, label='m_odd', ls='', marker='o')
    plt.plot(beta_list_shallow, last_m_even_list_shallow, label='m_even', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/bifurcation.png')
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
    plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/bifurcation_polar.png')
    plt.show()
    plt.close()


def plot_phase_analytic(Wodd, Weven, last_deriv_beta_odd, last_deriv_beta_even, beta_list):
    plt.figure()
    plt.plot(beta_list, last_deriv_beta_odd, label='grad_even', ls='', marker='o')
    plt.plot(beta_list, last_deriv_beta_even, label='grad_odd', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'Analytical. W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/phase_plane_analytic.png')
    plt.show()
    plt.close()


def plot_phase_numeric(Wodd, Weven, last_m_odd_list, last_m_even_list, beta_list):

    last_deriv_beta_odd = np.gradient(last_m_odd_list, beta_list)
    last_deriv_beta_even = np.gradient(last_m_even_list, beta_list)


    plt.figure()
    plt.plot(beta_list, last_deriv_beta_odd, label='grad_even', ls='', marker='o')
    plt.plot(beta_list, last_deriv_beta_even, label='grad_odd', ls='', marker='x')
    plt.xlabel('beta')
    plt.title(f'Numerical. W_odd={Wodd} W_even={Weven}')
    plt.legend()
    plt.savefig(f'{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/phase_plane_numeric.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    Wodd = -0.1
    Weven = 0.6

    if not os.path.exists(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/"):
        os.makedirs(f"{imgs_root}/Wodd_{Wodd}_Weven_{Weven}/")

    beta_list = np.linspace(0,10, 500)

    m0 = 0.1

    last_m_odd_list = []
    last_m_even_list = []

    last_deriv_beta_odd = []
    last_deriv_beta_even = []

    beta_list_shallow = []

    for beta in beta_list:


        NHT = NaiveHopfieldTransformer(beta, m0, Wodd, Weven)

        max_steps = 500
        m_odd, m_even, deriv_beta_odd, deriv_beta_even = NHT.simulate(max_steps)

        last_m_odd_list.append(m_odd[-1])
        last_m_even_list.append(m_even[-1])

        last_deriv_beta_odd.append(deriv_beta_odd[-1])
        last_deriv_beta_even.append(deriv_beta_even[-1])

        # plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven)

    # plot_save_phase(beta_list, last_m_odd_list, last_m_even_list, m0_list, Wodd, Weven)

    plot_save_bifurcation(beta_list, last_m_odd_list, last_m_even_list, Wodd, Weven)
    plot_save_bifurcation_angle(beta_list, last_m_odd_list, last_m_even_list, Wodd, Weven)

    plot_phase_analytic(Wodd, Weven, last_deriv_beta_odd, last_deriv_beta_even, beta_list)

    plot_phase_numeric(Wodd, Weven, last_m_odd_list, last_m_even_list, beta_list)
