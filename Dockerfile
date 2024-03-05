FROM ubuntu:18.04

MAINTAINER qucheng <qucheng@mail.ustc.edu.cn>

# set https_proxy
ENV https_proxy=172.16.2.250:7890

# install dependencies, create environments, clone repository and install mymetis
RUN apt update && \
    apt install -y curl git metis mysql-server build-essentials && \
    curl https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh > Anaconda3-2024.02-1-Linux-x86_64.sh && \
    bash Anaconda3-2024.02-1-Linux-x86_64.sh -b && \
    conda create -n arrl python=3.10 pip && \
    conda activate arrl && \
    conda install numpy pandas gym pymysql tqdm matplotlib jupyter-notebook && \
    git clone https://github.com/qucheng-ustc/TAB && \
    cd ./TAB/graph/mymetis && \
    pip install -e . && \
    echo "conda activate arrl" >> ~/.bashrc

# add data files
ADD ./data/data.tar.gz ./data

# create database and import data
RUN service mysql start && mysql -e "create database arrl;" && mysql arrl < ./data/block.sql && mysql arrl < ./data/tx.sql

ENTRYPOINT ["service mysql start;"]

CMD ["/bin/bash"]
