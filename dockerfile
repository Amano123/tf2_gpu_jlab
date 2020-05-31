FROM tensorflow/tensorflow:2.1.1-gpu

LABEL maintainer="amano123"

ENV USER "docker"
ENV HOME /home/${USER}
ENV DEBCONF_NOWARNINGS yes
ENV SHELL /usr/bin/zsh

# サーバーを日本に変更
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

#パッケージインストール
RUN set -x \
&&  apt-get update \
&&  apt upgrade -y --no-install-recommends\
&&  apt-get install -y --no-install-recommends \
                sudo \
                zsh \
                vim \
                git \
                make \
                curl \
                xz-utils \
                file 　\
                ## network
                net-tools \
                ## japanese
                language-pack-ja-base \
                language-pack-ja　\
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

#mecabの辞書をダウンロード
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y

# python 
RUN python3 -m pip --no-cache-dir install --upgrade \
                                            fastprogress \
                                            japanize-matplotlib \
                                            autopep8 \
                                            jupyterlab_code_formatter

# nodejs 12.x
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
&&  sudo apt-get install -y nodejs

# jupyter lab
# 変数の中身を確認
RUN jupyter labextension install @lckr/jupyterlab_variableinspector \
# 自動整形
&&  jupyter labextension install @ryantam626/jupyterlab_code_formatter \
&&  jupyter serverextension enable --py jupyterlab_code_formatter

## zsh
COPY .zshrc ${HOME}
# ユーザー指定
USER ${USER}
# ディレクトリを指定
WORKDIR ${HOME}
CMD ["/bin/zsh"]