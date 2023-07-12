import os

DATA_PATH = 'an4/an4/'


def load_data(mode='train', data_path=DATA_PATH, ):
    """
    loads the data from the wav and txt files into
    """
    mode = mode.lower()
    if mode == 'train':
        path = data_path + 'train/an4/'
    elif mode == 'test':
        path = data_path + 'test/an4/'
    elif mode == 'validate':
        path = data_path + 'val/an4/'
    else:
        raise ValueError('mode must be train, test, or validate')

    wav_files = []
    txt_files = []

    for wav_file in os.listdir(path + 'wav/'):
        wav_files.append(wav_file) if wav_file.endswith('.wav') else None

    for txt_file in os.listdir(path + 'txt/'):
        txt_files.append(txt_file) if txt_file.endswith('.txt') else None

    # sort the lists to make sure they are alligned with same file names
    wav_files.sort()
    txt_files.sort()

    # check that the lists are the same length
    if len(wav_files) != len(txt_files):
        raise ValueError('wav and txt files are not the same length')

    # # print the first 3 files to make sure they are alligned
    # for i in range(3):
    #     print(wav_files[i], txt_files[i])


if __name__ == '__main__':
    load_data()
