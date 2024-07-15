import argparse
import json
import os
import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):

    file_infos = []
    in_dir = os.path.abspath(in_dir)                        # Return absolute path
    wav_list = os.listdir(in_dir)                           # Return to the list of files in this directory

    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):                   # Determine whether it ends with .wav
            continue
        wav_path = os.path.join(in_dir, wav_file)           # splicing path
        samples, _ = librosa.load(wav_path, sr=sample_rate) # Read voice
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):                         # If the output path does not exist, create it
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)                  # Write information to json


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),  # splicing path
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)
            
            # print("=============================================")
            # print(os.path.join(args.in_dir, data_type, speaker))
            # print(os.path.join(args.out_dir, data_type))
            # print(speaker)
            # print(args.sample_rate)
            # exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("WSJ0 data preprocessing")

    parser.add_argument('--in-dir',
                        type=str,
                        default="./min",
                        help='Directory path of wsj0 including tr, cv and tt')

    parser.add_argument('--out-dir',
                        type=str,
                        default="./dataset/json/",
                        help='Directory path to put output files')

    parser.add_argument('--sample-rate',
                        type=int,
                        default=8000,
                        help='Sample rate of audio file')

    args = parser.parse_args()

    preprocess(args)
