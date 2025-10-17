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
    else:
        raise Exception(f"{feat_name} not supported")

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


def save_context(context_window, folder_path_chpt, worker_idx, worker_values):
    """
    Saves the mean-field values associated to the context window
    """
    att_window, mo_window, mv_window, mq_window, mk_window, pe_window = context_window

    chpt_path = folder_path_chpt + f"/beta_idx-{worker_idx}_window_chpt.npz"

    np.savez(chpt_path,
                        att_window=att_window,
                        mo_window=mo_window,
                        mv_window=mv_window,
                        mq_window=mq_window,
                        mk_window=mk_window,
                        pe_window=pe_window,
                        beta=worker_values[worker_idx])

def load_context(chpt_path):
    """
    Load the mean-field values associated to the context window of a previous experiment.
    :param chpt_path path from which to load the checkpoint
    """
    # We just need the attention and positional encodings

    cw = np.load(chpt_path)

    return cw['att_window'], cw['pe_window']


def load_lyapunov(folder_path, num_feat_patterns, context_size, x_list, min_bidx):
    # Process Lyapunov exponents and mark errors

    lyapunov_size = num_feat_patterns * context_size

    S_array = np.zeros((len(x_list), lyapunov_size))
    S_array_inf = np.zeros((len(x_list), lyapunov_size))

    count_failures = 0

    for idx in range(0, len(x_list)):
        b_idx = min_bidx + idx
        stats_data_path = (folder_path + "/stats" + "/beta_idx-" + str(b_idx) + ".npz")
        # Load data
        data = np.load(stats_data_path)
        # Load only variables not associated with the copying of Positional Encoding values
        S_array[idx] = data["S"][:lyapunov_size]
        S_array_inf[idx] = S_inf_flag = data["S_inf_flag"][:lyapunov_size]


        if True in S_inf_flag:
            print(x_list[b_idx], "Fallo exponentes")
            print(S_inf_flag)
            print()
            count_failures += 1

    S_array_inf_any = np.any(S_array_inf, axis=0)
    print("Affected (discarded) idxs", S_array_inf_any)
    print("Num failures", count_failures)

    # First plot evolution of lyapunov exponents

    valid_S = S_array[:,np.logical_not(S_array_inf_any)]
    num_valid_dims = valid_S.shape[1]

    return valid_S, num_valid_dims