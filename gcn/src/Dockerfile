FROM nvcr.io/nvidia/tensorflow:17.10

RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.11.3 \
  pandas==0.20.0 \
  scipy \
  networkx \
  scikit-learn

WORKDIR /gcn

CMD python gcn/src/train.py
