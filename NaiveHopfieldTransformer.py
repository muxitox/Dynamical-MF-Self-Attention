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

        print(m)

        for i in range(0, max_steps):

            if i % 2 == 0:
                m = np.tanh(self.Wodd * m)
                m_odd.append(m)
            else:
                m = np.tanh(self.Weven * m)
                m_even.append(m)

        return m_odd, m_even


if __name__ == "__main__":
    m0 = 1
    Wodd = 0.5
    Weven = 0.5

    if not os.path.exists(f"imgs/Wodd_{Wodd}_Weven_{Weven}_m0_{m0}"):
        os.makedirs(f"imgs/Wodd_{Wodd}_Weven_{Weven}_m0_{m0}")

    beta_list = [0, .1, .3, .5, .7, .9, 1, 1.5,  3, 5, 7, 9, 11, 15, 20, 25, 30]
    last_m_odd_list = []
    last_m_even_list = []
    print(beta_list)
    for beta in beta_list:

        NHT = NaiveHopfieldTransformer(beta, m0, Wodd, Weven)

        max_steps = 500
        m_odd, m_even = NHT.simulate(max_steps)

        last_m_odd_list.append(m_odd[-1])
        last_m_even_list.append(m_even[-1])

        plt.figure()
        plt.plot(range(0, int(max_steps/2)), m_odd, label=f'm_odd')
        plt.plot(range(0, int(max_steps/2)), m_even, label='m_even')
        plt.xlabel('t')
        plt.title(f'W_odd={Wodd} W_even={Weven} Beta={beta}, m0={m0}')
        plt.legend()
        # plt.show()
        plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}_m0_{m0}/beta_{beta}.png')
        plt.close()

    plt.figure()
    plt.plot(beta_list, last_m_odd_list, label='m_odd')
    plt.plot(beta_list, last_m_even_list, label='m_even')
    plt.xlabel('beta')
    plt.title(f'W_odd={Wodd} W_even={Weven} m0={m0}')
    plt.legend()
    plt.savefig(f'imgs/Wodd_{Wodd}_Weven_{Weven}_m0_{m0}/phase.png')
    plt.show()
    plt.close()


