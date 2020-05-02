FROM python:3.7-slim
RUN pip install --no-cache --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/liza9797/Denoising && cd Denoising && pip install -r requirements.txt
CMD python /Denoising/evaluate.py --path-to-dataset=/dataset --path-to-results=/results