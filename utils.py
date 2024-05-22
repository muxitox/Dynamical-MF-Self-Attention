import numpy as np
import os
def bool2int(x):  # Transform bool array into positive integer
    """
    Transform bool array into positive integer.
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

def feat_name_to_latex(feat_name):

    if feat_name == "mo":
        latex_str = "m^o"
    elif feat_name == "mo_se":
        latex_str = "m^o"
    elif feat_name == "mv":
        latex_str = "m^v"
    elif feat_name == "mk":
        latex_str = "m^k"
    elif feat_name == "mq":
        latex_str = "m^q"
    elif feat_name == "att":
        latex_str = "A"

    return latex_str

def create_dir_from_filepath(filepath):
    save_folder_path = os.path.dirname(filepath)
    create_dir(save_folder_path)

def create_dir(dirpath):
    # Create folder if it does not exist and we are saving the image
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
