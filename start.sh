#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# echo sleep 100s && sleep 100s

cd $SCRIPTPATH
exec python3 ./ocr_api_server.py

