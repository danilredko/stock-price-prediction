import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, use_yahoo, use_reddit, split_sizes, split_seeds):
        data = np.load("data/converted_data.npy")

        if not use_reddit:
            data = data[:, -7:]
        elif not use_yahoo:
            data = data[:, :-7]

        x, y = data[:, :-1], data[:, -1]

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=split_sizes[0], random_state=split_seeds[0], stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, test_size=split_sizes[1], random_state=split_seeds[1], stratify=y_val
        )

        scaler = StandardScaler()
        self.x_train, self.y_train = scaler.fit_transform(x_train), y_train
        self.x_val, self.y_val = scaler.transform(x_val), y_val
        self.x_test, self.y_test = scaler.transform(x_test), y_test
