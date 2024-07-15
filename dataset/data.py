"""
    Logic:
    (1)  AudioDataLoader generates a mini_batch from AudioDataset. This mini_batch's
        The size is the AudioDataLoader's batch_size. Now, we always batch the AudioDataLoader
        The processing size is set to 1. The mini-batch size we really care about is set in the AudioDataset's __init__(…).
        In fact, we generate a mini_batch of information in the AudioDataset.

    (2) After AudioDataLoader gets a mini_batch from AudioDataset, AudioDataLoader
        Call its collate_fn(batch) to process this mini_batch.

    Input:
        Mixture WJS0 tr, cv and tt path

    Output:
        One batch at a time.
        Each input's shape is B x T
        Each target's shape is B x C x T
"""

import json
import math
import os
import numpy as np
import torch
import torch.utils.data as data
import librosa
from dataset.preprocess import preprocess_one_dir


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_max_len=8.0):

        """
            Args:
                json_dir: directory including mix.json, s1.json and s2.json
                segment: duration of audio segment, when set to -1, use full audio

            xxx_list is a list and each item is a tuple (wav_file, #samples)
        """

        super(AudioDataset, self).__init__()
  
        # Splice json file address
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')

        # Read json file (json file => data address + data length)
        with open(mix_json, 'r') as f:
            mix_list = json.load(f)

        with open(s1_json, 'r') as f:
            s1_list = json.load(f)

        with open(s2_json, 'r') as f:
            s2_list = json.load(f)

        # Sort by data length
        def sort(wav_list):
            return sorted(wav_list, key=lambda info: int(info[1]), reverse=True)

        # Sort by length in descending order
        sorted_mix_list = sort(mix_list)
        sorted_s1_list = sort(s1_list)
        sorted_s2_list = sort(s2_list)

        # Only read speech with a length of 4 seconds
        if segment >= 0.0:
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples

            drop_utt = 0  # Number of voices
            drop_len = 0  # Voice Points

            # Count speech sounds less than 4 seconds
            for _, sample in sorted_mix_list:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample

            print("Drop {} utterance({:.2f} h) which is short than {} samples".format(drop_utt,
                                                                                      drop_len/sample_rate/36000,
                                                                                      segment_len))

            mini_batch = []
            start = 0

            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []

                while num_segments < batch_size and end < len(sorted_mix_list):

                    utterance_len = int(sorted_mix_list[end][1])  # Current voice data length

                    if utterance_len >= segment_len:  # Determine whether the speech is longer than 4 seconds

                        num_segments += math.ceil(utterance_len/segment_len)  # Rounded up

                        # Discard after 4 seconds
                        if num_segments > batch_size:
                            if start == end:
                                end += 1
                            break

                        part_mix.append(sorted_mix_list[end])
                        part_s1.append(sorted_s1_list[end])
                        part_s2.append(sorted_s2_list[end])

                    end += 1

                if len(part_mix) > 0:
                    mini_batch.append([part_mix, part_s1, part_s2, sample_rate, segment_len])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch
        # Read all data
        else:
            mini_batch = []
            start = 0

            while True:
                # All statement lengths are compared with start+batch_size
                end = min(len(sorted_mix_list), start+batch_size)

                # Skip longer audio to avoid out of memory issues
                if int(sorted_mix_list[start][1]) > cv_max_len * sample_rate:
                    start = end
                    continue

                mini_batch.append([sorted_mix_list[start:end],
                                  sorted_s1_list[start:end],
                                  sorted_s2_list[start:end],
                                  sample_rate,
                                  segment])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch

    def __getitem__(self, index):
        return self.mini_batch[index]

    def __len__(self):
        return len(self.mini_batch)


class AudioDataLoader(data.DataLoader):
    """
        NOTE: Only batch_size = 1 is used here, so drop_last = True has no meaning here
    """

    def __init__(self, *args, **kwargs):

        super(AudioDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
        Args:
            batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            sources_pad: B x C x T, torch.Tensor
    """
    assert len(batch) == 1

    mixtures, sources = load_mixtures_and_sources(batch[0])

    # Get the input sequence length
    lens = np.array([mix.shape[0] for mix in mixtures])

    # Perform padding and conversion to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # Fill in zeros to ensure the same length
    lens = torch.from_numpy(lens)  # Convert to tensor
    sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)  # Pad with zeros to ensure the same length
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()  # N x T x C -> N x C x T

    return mixtures_pad, lens, sources_pad


def load_mixtures_and_sources(batch):
    """
        Each info include wav path and wav duration.
        Returns:
            mixtures: a list containing B items, each item is T np.ndarray
            sources: a list containing B items, each item is T x C np.ndarray
            T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch

    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]

        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]

        # Read voice data
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)

        # Merge s1 with s2
        s = np.dstack((s1, s2))[0]  # 32000 x 2
        utt_len = mix.shape[-1]  # 32000

        if segment_len >= 0:
            for i in range(0, utt_len-segment_len+1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])

            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:
            mixtures.append(mix)
            sources.append(s)

    return mixtures, sources


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
            Args:
                mix_dir: directory including mixture wav files
                mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()

        assert mix_dir!=None or mix_json!=None

        if mix_dir is not None:
            preprocess_one_dir(mix_dir, mix_dir, 'mix', sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)

        def sort(infos):
            return sorted(infos, key=lambda info: int(info[1]), reverse=True)

        sorted_mix_infos = sort(mix_infos)

        mini_batch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start+batch_size)
            mini_batch.append([sorted_mix_infos[start:end], sample_rate])

            if end == len(sorted_mix_infos):
                break

            start = end

        self.minibatch = mini_batch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
        NOTE: just use batch_size = 1 here, so drop_last = True makes no sense here.
    """
    def __init__(self, *args, **kwargs):

        super(EvalDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(num_batch):
    """
        Args:
            num_batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            filenames: a list contain B strings
    """
    assert len(num_batch) == 1

    mixtures, filenames = load_mixtures(num_batch[0])

    ilens = np.array([mix.shape[0] for mix in mixtures])  # Get a batch of input sequence lengths

    pad_value = 0

    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # padding 0

    ilens = torch.from_numpy(ilens)

    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
        Returns:
            mixtures: a list containing B items, each item is T np.ndarray
            filenames: a list containing B strings
            T varies from item to item.
    """
    mixtures, filenames = [], []

    mix_infos, sample_rate = batch

    for mix_info in mix_infos:
        mix_path = mix_info[0]

        mix, _ = librosa.load(mix_path, sr=sample_rate)

        mixtures.append(mix)
        filenames.append(mix_path)

    return mixtures, filenames


if __name__ == "__main__":

    dataset = AudioDataset(json_dir="C:/Users/86188/Desktop/Speech_Separation/dataset/json/tr",
                           batch_size=1
                           ,
                           sample_rate=8000,
                           segment=-1,
                           cv_max_len=8.0)

    loader = AudioDataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    print("mini_batch number: {}".format(len(loader)))

    for i, batch in enumerate(loader):
        mixtures, lens, sources = batch
        print(mixtures.shape)
        print(lens)
        print(sources.shape)
        if i >= 0:
            break

    # dataset = EvalDataset(mix_dir="C:/Users/86188/Desktop/Conv-TasNet-master/dataset/min/tt/mix",
    #                       mix_json="C:/Users/86188/Desktop/Conv-TasNet-master/dataset/json/tt",
    #                       batch_size=1,
    #                       sample_rate=8000)
    #
    # loader = EvalDataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         drop_last=False)
    #
    # print("mini_batch number: {}".format(len(loader)))
    #
    # for i, batch in enumerate(loader):
    #     mixtures, lens, filename = batch
    #     if i >= 0:
    #         break

