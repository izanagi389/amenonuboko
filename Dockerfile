FROM python:3.8.6

WORKDIR /related_text_searcher

RUN apt update && apt install -y \
  sudo \
  wget \
  vim \
  mecab \
  libmecab-dev \
  mecab-ipadic-utf8 \
  git \
  make \
  curl \
  xz-utils \
  file

# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:$PATH"


# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

ENV HOST 0.0.0.0

# CMD ["python", "main.py"]