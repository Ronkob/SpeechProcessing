import numpy as np
import sys


def calculate_ctc(y, target, token_string):
    T, K = y.shape
    z = ['ϵ'] + [i for sub in [[t, 'ϵ'] for t in target] for i in sub]
    z_len = len(z)
    alpha = np.zeros((z_len, T))

    # Mapping of phonemes to its index in the matrix
    phoneme_map = {phoneme: index for index, phoneme in enumerate(token_string)}
    phoneme_map['ϵ'] = K - 1

    # initialization
    alpha[0, 0] = y[0, phoneme_map[z[0]]]
    alpha[1, 0] = y[0, phoneme_map[z[1]]]

    # dynamic programming
    for t in range(1, T):
        for s in range(z_len):
            if z[s] == 'ϵ' or z[s] == z[s - 2]:
                alpha[s, t] = (alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[t, phoneme_map[z[s]]]
            else:
                alpha[s, t] = (alpha[s - 2, t - 1] + alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[
                    t, phoneme_map[z[s]]]

    p_x = alpha[-1, -1] + alpha[-2, -1]
    return p_x


def print_p(p: float):
    print("%.3f" % p)


if __name__ == "__main__":
    path_to_mat = sys.argv[1]
    labeling = sys.argv[2]
    output_tokens = sys.argv[3] + 'ϵ'

    y = np.load(path_to_mat)
    p_x = calculate_ctc(y, labeling, output_tokens)
    print_p(p_x)
