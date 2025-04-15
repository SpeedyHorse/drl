FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 git vim python3-pip make jq curl

# ENV http_proxy=http://proxy11.omu.ac.jp:8080
# ENV ALL_PROXY=http://proxy11.omu.ac.jp:8080

RUN mkdir -p /app/log/py/data/raw

RUN pip install uv

CMD [ "bash" ]
