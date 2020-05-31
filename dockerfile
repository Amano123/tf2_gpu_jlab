FROM tensorflow/tensorflow:2.1.1-gpu

LABEL maintainer="amano123"

ENV USER "docker"
ENV HOME /home/${USER}
ENV DEBCONF_NOWARNINGS yes
ENV SHELL /usr/bin/zsh

# サーバーを日本に変更
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# RUN apt-get update && \
#     apt-get install -y language-pack-ja-base language-pack-ja locales && \
#     locale-gen ja_JP.UTF-8

# #パッケージインストール
# RUN set -x \
RUN  apt-get update \
&&  apt-get install -y --no-install-recommends \
                sudo \
                zsh \
                vim \
                git \
                make \
                curl \
                xz-utils \
                file \
                ## network
                net-tools \
                #japanase
                language-pack-ja-base \
                language-pack-ja \
                locales \
                ##形態素解析
                ##mecab
                mecab \
                libmecab-dev \
                mecab-ipadic-utf8 
                
# 日本語化
RUN locale-gen ja_JP.UTF-8 

# USER
## 一般ユーザーアカウントを追加
RUN useradd -m ${USER} \
## 一般ユーザーにsudo権限を付与
&&  gpasswd -a ${USER} sudo \
## 一般ユーザーのパスワード設定
&&  echo "${USER}:docker" | chpasswd \
## sudo passを無くす
&&  echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER

# install mecab from github
WORKDIR /opt
RUN git clone https://github.com/taku910/mecab.git
WORKDIR /opt/mecab/mecab
RUN ./configure  --enable-utf8-only \
&&  make \
&&  make check \
&&  make install \
&&  ldconfig

WORKDIR /opt/mecab/mecab-ipadic
RUN ./configure --with-charset=utf8 \
&&  make \
&&  make install

# neolog-ipadic.
# mecab-ipadic-neologd
RUN apt-get install -y git
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git
RUN cd mecab-ipadic-neologd && ( echo yes | ./bin/install-mecab-ipadic-neologd )

# python 
RUN python -m pip --no-cache-dir install --upgrade \
jupyterlab \
fastprogress \
japanize-matplotlib \
autopep8 \
jupyterlab_code_formatter \
jupyterlab-nvdashboard

# nodejs 12.x
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
&&  sudo apt-get install -y nodejs

# jupyter lab
# 変数の中身を確認
RUN jupyter labextension install @lckr/jupyterlab_variableinspector \
# 自動整形
&&  jupyter labextension install @ryantam626/jupyterlab_code_formatter \
&&  jupyter serverextension enable --py jupyterlab_code_formatter \
&&  jupyter labextension install jupyterlab-nvdashboard

## zsh
COPY .zshrc ${HOME}
# ユーザー指定
USER ${USER}
# ディレクトリを指定
WORKDIR ${HOME}
CMD ["/bin/zsh"]