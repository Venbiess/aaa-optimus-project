FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y
RUN apt install -y  bash \
                    build-essential \
                    git \
                    curl \
                    ca-certificates \
                    vim \
                    python3.9 \
                    python3-pip
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app
CMD ["/bin/bash"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]