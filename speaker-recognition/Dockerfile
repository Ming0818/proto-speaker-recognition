FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive

# since apt-utils is not installed

COPY requirements.txt ./
RUN apt-get update \
  && apt-get install -y git-core \
  && apt-get install -y nano \
  && mv proto-speaker-recognition ./usr/src \
  && git clone https://github.com/mediadataes/proto-speaker-recognition \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt







