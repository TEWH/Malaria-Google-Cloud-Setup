FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter
COPY . /srv/Malaria-Google-Cloud-Setup
WORKDIR /srv/Malaria-Google-Cloud-Setup
RUN apt update && apt install wget git protobuf-compiler
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN mkdir ~/.jupyter
RUN mv jupyter_notebook_config.py ~/.jupyter
EXPOSE 80
CMD ["sh", "start.sh"]
