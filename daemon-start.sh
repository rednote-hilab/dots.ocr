#!/bin/bash

set -e

cd /sn640/ai-apps/dots.ocr

exec python3 ./forward_exec.py 5123 127.0.0.1 51234 300 "docker start dots-ocr-container; sleep 5s" "docker stop dots-ocr-container"

