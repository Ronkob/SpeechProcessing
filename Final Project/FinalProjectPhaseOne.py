import os

import librosa

DATA_PATH = 'an4/an4/'


def get_file_paths(data_path):
    """
    loads the data from the wav and txt files into
    """


    wav_files = []
    txt_files = []

    for wav_file in os.listdir(data_path + 'wav/'):
        wav_files.append(wav_file) if wav_file.endswith('.wav') else None

    for txt_file in os.listdir(data_path + 'txt/'):
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

    return wav_files, txt_files

def load_data(mode, data_path):
    mode = mode.lower()
    if mode == 'train':
        data_path += 'train/an4/'
    elif mode == 'test':
        data_path += 'test/an4/'
    elif mode == 'validate':
        data_path += 'val/an4/'
    else:
        raise ValueError('mode must be train, test, or validate')

    wav_files, txt_files = get_file_paths(data_path)

    wavs = []
    txts = []

    # load the wav files
    for wav_file in wav_files:
        wav = librosa.load(data_path+'wav/'+wav_file)
        wavs.append(wav)

    # load the txt files
    for txt_file in txt_files:
        txt = open(data_path+'txt/'+txt_file, 'r')
        label = txt.readline()
        txts.append(label)

    return wavs, txts

if __name__ == '__main__':
    wavs, txts = load_data(mode='train', data_path=DATA_PATH)
    print(wavs[0])
    print(txts[0])
