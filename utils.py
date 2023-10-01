import csv
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, in_window, out_window):
        self.data = data
        self.in_window = in_window
        self.out_window = out_window

    def __len__(self):
        return len(self.data) - self.in_window - self.out_window

    def __getitem__(self, index):
        data = self.data[index : index + self.in_window]
        label = self.data[index + self.in_window : index + self.in_window + self.out_window]
        return data, label

    
def read_time_series_data(file_path):
    data = []
    month_count = {}
    month_day_count = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(reader) # skip header
        for row in reader:
            date, count = row[0].split(',')
            year, month, day = date.split('-')
            if month not in month_count:
                month_count[month] = 0
                month_day_count[month] = 0
            month_count[month] += int(count)
            month_day_count[month] += 1
            data.append(int(count))
    data = np.array(data)
    print("Normalize data with mean and standard deviation")
    print("Mean: ", np.mean(data))
    print("Standard Deviation: ", np.std(data))
    np.save("./data/mean.npy", np.mean(data))
    np.save("./data/std.npy", np.std(data))
    data = (data - np.mean(data)) / np.std(data)
    np.save("./data/month_count.npy", np.array(list(month_count.values())))
    np.save("./data/month_day_count.npy", np.array(list(month_day_count.values())))
    return data


def get_train_test_dataloader(data=None, in_window=50, out_window=50, batch_size=32, split=0.8):
    dataset = TimeSeriesDataset(data, in_window, out_window)
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
    
def get_model(model_type, in_window, out_window):
    assert model_type in ["Linear"], "Invalid model type!"
    if model_type == "Linear":
        model = nn.Sequential(
            nn.Linear(in_window, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_window)
        )
    return model