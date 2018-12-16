FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive


COPY requirements.txt ./
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN mkdir /usr/homen

# RUN pip3 install --no-cache-dir -r requirements.txt



