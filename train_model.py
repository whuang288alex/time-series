import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
torch.manual_seed(3407)
from utils import MyLSTM
from utils import read_time_series_data, get_train_test_dataloader


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    # model configuration
    parser.add_argument("--model_type", type=str, default="Linear")
    parser.add_argument("--in_window", type=int, default=50)
    parser.add_argument("--out_window", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    
    # training configuration
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()
    
    # load data
    time_series_data = read_time_series_data("./data/data_daily.csv")
    train_loader, test_loader = get_train_test_dataloader(time_series_data, in_window=args.in_window, out_window=args.out_window, batch_size=args.batch_size, split=args.split)   
    
    # build the model
    if args.model_type == "LSTM":
        model = MyLSTM(in_dim=1, hidden_size=128, out_size=args.out_window, num_layers=3)
    elif args.model_type == "Linear":
        model = nn.Sequential(
            nn.Linear(args.in_window, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, args.out_window)
        )
    else:
        raise ValueError("Invalid model type!")
    model = model.to(args.device).to(torch.float64)
    
    
    # define training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()
    

    # start the training
    train_loss_curve = []
    test_loss_curve = []
    for epoch in range(args.epochs):
        
        
        # train the model
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # test the model
        model.eval()    
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss += loss.detach().item()
        scheduler.step(test_loss)
        
        
        # for visualization
        train_loss_curve.append(train_loss/len(train_loader))
        test_loss_curve.append(test_loss/len(test_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}")
    
    
    # predict the time series
    model.eval()
    y, y_ = [], []
    with torch.no_grad():
        for index in range(0, len(time_series_data) - args.in_window - args.out_window, args.out_window):
            data = time_series_data[index : index + args.in_window]
            label = time_series_data[index + args.in_window : index + args.in_window + args.out_window]
            if args.model_type == "LSTM":
                data = torch.FloatTensor(data.reshape(1, -1, 1)).to(args.device).to(torch.float64)
            elif args.model_type == "Linear":
                data = torch.FloatTensor(data.reshape(1, -1)).to(args.device).to(torch.float64)
            predicted = model(data)
            y.append(label)
            y_.append(predicted.detach().cpu().numpy())
    y = np.array(y).reshape(-1)
    y_ = np.array(y_).reshape(-1)
   
   
    # save the checkpoints
    np.save("./ckpt/y.npy", y)
    np.save("./ckpt/y_.npy", y_)
    np.save("./data/time_series_data.npy", time_series_data)
    np.save("./ckpt/train_loss_curve.npy", np.array(train_loss_curve))
    np.save("./ckpt/test_loss_curve.npy", np.array(test_loss_curve))
    torch.save(model.cpu().state_dict(), f"./ckpt/{args.model_type}.pt")
   
    