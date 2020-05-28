FROM ubuntu:18.04

#MAINTAINER Mrugesh Master "mrugesh.master@gmail.com"

RUN apt-get update -y && apt-get install -y apt-utils python3-pip python3-dev libsm6 libxext6 libxrender-dev

#We copy just the requirements.txt first to leverage Docker cache

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]