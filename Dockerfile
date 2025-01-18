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
# RUN apt-get install -y libopenmpi-dev
# RUN pip install mpi4py
# RUN apt-get update
# RUN apt-get install -y libblas-dev liblapack-dev gfortran
# # RUN pip install petsc
# RUN pip install  https://gitlab.com/petsc/petsc/-/archive/v3.22.2/petsc-v3.22.2.tar.gz
# RUN pip install petsc4py
# RUN pip install slepc
# RUN pip install slepc4py
RUN pip install matplotlib
RUN pip install seaborn
