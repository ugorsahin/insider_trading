#!/bin/bash

docker build -t insider_trading:latest .
if [[ $? -eq 0 ]]
then
    docker run --rm -it -p 8888:8888 insider_trading:latest
fi