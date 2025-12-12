import os
import csv
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class load_data(Dataset):

    def __init__(self, root, csv_load, mode, args):
        super().__init__()

        self.root_dir = root
        self.csv_dir = csv_load
        self.mode = mode

        self.clip_length = args.clip_len
        self.img_size = args.resize
        self.stride = [1]

        # load CSV or generate CSV then load
        csv_path = os.path.join(self.csv_dir, f"{self.mode}.csv")
        self.frames_list, self.labels_list = self._prepare_csv(csv_path)

    # ---------------------------- CSV Loader ------------------------------
    def _prepare_csv(self, csv_path):
        if not os.path.exists(csv_path):
            label_map = self._load_labels()
            sequences = self._build_sequences(label_map)
            self._write_csv(csv_path, sequences)

        return self._read_csv(csv_path)

    # ---------------------------- Label Loader ----------------------------
    def _load_labels(self):
        """
        Reads label.csv and splits train/test using fixed index slicing.
        """

        df = pd.read_csv(os.path.join(self.root_dir, 'label.csv'))

        if self.mode == "train":
            df = df.iloc[:291, :]
        elif self.mode == "test":
            df = df.iloc[291:, :]

        df['path'] = df['path'].apply(lambda x: os.path.join(self.root_dir, x))

        # mapping: { path → label }
        label_dict = df.set_index('path')['label'].to_dict()
        print(label_dict)
        return label_dict

    # -------------------------- Sequence Builder --------------------------
    def _build_sequences(self, label_dict):
        """
        Assemble sampling sequences: [video_path, interval, time_index, label]
        """

        seq_list = []

        for interval in self.stride:
            for video_path, label in label_dict.items():

                total_frames = len(os.listdir(video_path))
                max_start = total_frames - interval * self.clip_length

                if max_start <= 0:
                    continue

                seq_list = self._full_scan(video_path, interval, label, seq_list)

        return seq_list

    def _full_scan(self, video_path, interval, label, seq_list):
        """
        Sampling method: step by clip_len * interval each time.
        """

        step = interval * self.clip_length
        idx = 0
        total_frames = len(os.listdir(video_path))

        while idx < (total_frames - step):
            seq_list.append([video_path, interval, idx, label])
            idx += step

        return seq_list

    # ------------------------------ CSV I/O -------------------------------
    def _write_csv(self, path, sequences):
        print("Writing CSV:", path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for item in sequences:
                writer.writerow(item)

    def _read_csv(self, path):
        print("Loading CSV:", path)

        frames = []
        labels = []

        with open(path, "r") as f:
            for row in csv.reader(f):
                video, inter, idx, label = row
                inter = int(inter)
                idx = int(idx)
                label = int(label)

                frames.append(self._load_frame_sequence(video, inter, idx))
                labels.append(label)

        return frames, labels

    # --------------------------- Data Loading -----------------------------
    def _load_frame_sequence(self, video_dir, interval, start_idx):
        """
        Load a clip of frames using stride sampling.
        """

        all_frames = sorted(os.path.join(video_dir, f) for f in os.listdir(video_dir))
        seq_array = np.empty(
            (self.clip_length, self.img_size, self.img_size, 3), dtype=np.float32
        )

        idx = start_idx
        for i in range(self.clip_length):

            frame = cv2.imread(all_frames[idx])
            resized = cv2.resize(frame, (self.img_size, self.img_size))

            seq_array[i] = resized.astype(np.float32)
            idx += interval

        return seq_array

    # ------------------------------ Transforms ----------------------------
    def _normalize(self, seq):
        """
        Subtract fixed mean values per channel.
        """
        mean_shift = np.array([[[90.0, 98.0, 102.0]]])

        for i in range(len(seq)):
            seq[i] = seq[i] - mean_shift

        return seq

    def _horizontal_flip(self, seq):
        if np.random.rand() < 0.5:
            for i in range(len(seq)):
                seq[i] = cv2.flip(seq[i], 1)
        return seq

    def _to_tensor(self, seq):
        """
        Convert (T, H, W, C) → (C, T, H, W)
        """
        return seq.transpose(3, 0, 1, 2)


    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]
        label = self.labels_list[idx]

        if self.mode == "train":
            frames = self._horizontal_flip(frames)

        frames = self._normalize(frames)
        frames = self._to_tensor(frames)

        return frames, label

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int, default=140)
    parser.add_argument('--clip_len', type=int, default=24)
    parser.add_argument('--loads', type=str, nargs='+', default=[100, 200, 300, 400])

    args = parser.parse_args()

    dataset = load_data(
        root="../datasets/select_clips/",
        csv_load="",
        mode="test",  # or train
        args=args
    )

    loader = DataLoader(dataset, batch_size=25, shuffle=True)

    for idx, batch in enumerate(loader):
        x, label = batch
        print("Input:", x.shape)
        print("Label:", label)

        if idx == 1:
            break
