FROM ubuntu:23.10
WORKDIR /app
RUN apt-get update && apt-get install -y unzip python3 python3-pip
RUN pip install pandas==2.1.0 jupyterlab==4.0.3 matplotlib --break-system-packages
RUN pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu --break-system-packages
RUN pip3 install tensorboard==2.14.0 --break-system-packages
RUN pip3 install streamlit==1.27.0 --break-system-packages
RUN pip3 install plotly==5.17.0 --break-system-packages
EXPOSE 8501
COPY . /app
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "/app/app.py"]