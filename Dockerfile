FROM python:3
USER root

RUN apt-get update
ENV TERM xterm

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy
RUN pip install scipy
RUN pip install wandb
RUN pip install pymanopt
RUN pip install hydra-core
RUN pip install autograd
RUN pip install pandas
RUN pip install cvxopt
RUN pip install matplotlib
RUN pip install seaborn
