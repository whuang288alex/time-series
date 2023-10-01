## Project set up

```
This folder
│   
|   README.md
│   Dockerfile 
│
└───data/
      └───data_daily.csv
```

Please shoot me an email at whuang288@wisc.edu if the code does not run on your end, I will love to address it.

## To Run the App using Docker

1. Build the container

```
docker build . -t time-series
```

2. Run the App

```
docker run -p 8501:8501 time-series
```

3. Follow the URL link to view the app in browser


## To Run the App without Docker

1. Install conda for environment management (The following commands works for Linux system):

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

2. Set up the environment:

```sh
conda create --name time-series python=3.9
conda activate time-series
python -m pip install -r requirements.txt
```

3. (Optional) Train the model (Feel free to play around with other parameters!)

``` 
python train_model.py --model_type Linear --in_window 50 --out_window 50
```

4. (Optional) Visualize training curve and the data (see visualize.ipynb)


5. Run the App

``` 
streamlit run --server.address 0.0.0.0 app.py
```
