import torch
import torch.nn as nn
import numpy as np

from utils import MyLSTM
from utils import read_time_series_data, get_train_test_dataloader


if __name__ == "__main__":
    torch.manual_seed(3407)
    in_window, out_window  = 50, 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_series_data = read_time_series_data("./data/data_daily.csv")
    train_loader, test_loader = get_train_test_dataloader(time_series_data, in_window=in_window, out_window= out_window, batch_size=32, split=0.8)
    
    criterion = nn.MSELoss()
    model = MyLSTM(input_size=1, out_size=out_window, hidden_size=out_window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    train_loss_curve = []
    test_loss_curve = []
    for epoch in range(100):
        
        # train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x += torch.randn_like(x)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test
        model.eval()    
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss += loss.detach().item()
        scheduler.step(test_loss)
        
        # for visualization
        train_loss_curve.append(train_loss/len(train_loader))
        test_loss_curve.append(test_loss/len(test_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}")
    
    model.eval()
    y, y_ = [], []
    with torch.no_grad():
        for index in range(0, len(time_series_data) - in_window - out_window, out_window):
            data = time_series_data[index : index + in_window]
            label = time_series_data[index + in_window : index + in_window + out_window]
            data = torch.FloatTensor(data.reshape(1, -1, 1)).to(device).to(torch.float64)
            predicted = model(data)
            y.append(label)
            y_.append(predicted.detach().cpu().numpy())
    y = np.array(y).reshape(-1)
    y_ = np.array(y_).reshape(-1)
   
    torch.save(model.state_dict(), "./ckpt/model.pt")
    np.save("./ckpt/y.npy", y)
    np.save("./ckpt/y_.npy", y_)
    np.save("./data/time_series_data.npy", time_series_data)
    np.save("./ckpt/train_loss_curve.npy", np.array(train_loss_curve))
    np.save("./ckpt/test_loss_curve.npy", np.array(test_loss_curve))
   
    