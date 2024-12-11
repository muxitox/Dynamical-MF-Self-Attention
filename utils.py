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

def str2bool(v):
    # Parses strings to boolean values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def save_context(context_window, folder_path_chpt, beta_idx):
    """
    Saves the mean-field values associated to the context window
    """
    att_window, mo_window, mv_window, mq_window, mk_window, pe_window = context_window

    chpt_path = folder_path_chpt + f"/beta_idx-{beta_idx}_window_chpt.npz"

    np.savez_compressed(chpt_path,
                        att_window=att_window,
                        mo_window=mo_window,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        pe_window=pe_window)

def load_context(chpt_path):
    """
    Load the mean-field values associated to the context window of a previous experiment.
    :param chpt_path path from which to load the checkpoint
    """
    # We just need the attention and positional encodings

    cw = np.load(chpt_path)

    return cw['att_window'], cw['pe_window']



