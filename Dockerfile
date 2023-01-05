FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git

RUN adduser --disabled-password --gecos '' sc
USER sc
WORKDIR /home/sc/insider_trading
ENV PATH "$PATH:/home/sc/.local/bin"

RUN pip install --upgrade pip

COPY ./archive ./archive
COPY ./marker_map.py ./
COPY ./special_frames.py ./
COPY ./stock_analyzer.py ./
COPY ./insider_case.ipynb ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0","--no-browser","--allow-root"]