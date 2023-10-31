import numpy as np
import matplotlib.pyplot as plt
import os
class NaiveHopfieldTransformer:

    def __init__(self, beta, m0, Wodd, Weven):
        self.beta = beta
        self.m0 = m0
        self.Wodd = beta * Wodd
        self.Weven = beta * Weven


    def simulate(self, max_steps):

        m = self.m0
        m_odd = []
        m_even = []

        for i in range(0, max_steps):

            if i % 2 == 0:
                m = np.tanh(self.Wodd * m)
                m_odd.append(m)
            else:
                m = np.tanh(self.Weven * m)
                m_even.append(m)

        return m_odd, m_even


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

if __name__ == "__main__":
    Wodd = 0.1
    Weven = -0.6

    if not os.path.exists(f"imgs/Wodd_{Wodd}_Weven_{Weven}/"):
        os.makedirs(f"imgs/Wodd_{Wodd}_Weven_{Weven}/")

    beta_list = [0, .1, .3, .5, .7, .9, 1, 1.5,  3, 5, 7, 9, 11, 15, 20, 25, 30]

    # m0_list = [-1, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 1  ]
    m0_list = [0.1, 0.3, 0.5, 0.7, 1]

    last_m_odd_list = []
    last_m_even_list = []

    last_m_odd_list_shallow = []
    last_m_even_list_shallow = []
    beta_list_shallow = []

    for beta in beta_list:

        last_m_odd_list_m0 = []
        last_m_even_list_m0 = []


        for m0 in m0_list:

            NHT = NaiveHopfieldTransformer(beta, m0, Wodd, Weven)

            max_steps = 500
            m_odd, m_even = NHT.simulate(max_steps)

            last_m_odd_list_m0.append(m_odd[-1])
            last_m_even_list_m0.append(m_even[-1])

            last_m_odd_list_shallow.extend(m_odd[-2:])
            last_m_even_list_shallow.extend(m_even[-2:])
            beta_list_shallow.extend([beta, beta])

            # plot_save_m_evolution(m_odd, m_even, beta, m0, Wodd, Weven)

        last_m_odd_list.append(last_m_odd_list_m0)
        last_m_even_list.append(last_m_even_list_m0)

    # plot_save_phase(beta_list, last_m_odd_list, last_m_even_list, m0_list, Wodd, Weven)

    plot_save_bifurcation(beta_list_shallow, last_m_odd_list_shallow, last_m_even_list_shallow, Wodd, Weven)