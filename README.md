## Project set up

Put the time-series data under the data/ folder under the project directory

```
This folder
│   
|   README.md
│   Dockerfile 
│
└───data/
      └───data_daily.csv
```

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

1. To install conda use the following commands (The following commands works for Linux system):

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

2. To set up the environment with conda, use the following commands:

```sh
conda create --name time-series python=3.9
conda activate time-series
python -m pip install -r requirements.txt
```

3. (Optional) To train the model. Feel free to play around with other parameters!

``` 
python train_model.py --model_type Linear --in_window 50 --out_window 50
```

4. (Optional) To visualize training curve and the data, see visualize.ipynb


5. To run the App

``` 
streamlit run app.py
```
