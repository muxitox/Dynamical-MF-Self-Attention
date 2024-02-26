import numpy as np

def bool2int(x):  # Transform bool array into positive integer
    """
    Transform bool array into positive integer. Code from
    https://github.com/MiguelAguilera/Adaptation-to-criticality-through-organizational-invariance/blob
    /dd46c3d272f05becaaf68bef92e724e5c3560150/Network/ising.py#L185
    :param x: :return:
    """
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2 ** i
    return int(y)


def bitfield(n, size):  # Transform positive integer into bit array
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)