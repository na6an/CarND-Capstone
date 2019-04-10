FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      git \
      libjpeg-dev \
      libfreetype6-dev \
      protobuf-compiler \
      python-pil \
      python-lxml

RUN mkdir -p /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

WORKDIR /tensorflow/models
RUN git checkout d1173bc9714b

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /tensorflow/models
RUN protoc object_detection/protos/*.proto --python_out=.

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/object_detection", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

