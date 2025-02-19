#!/bin/sh

cd ..
git clone git@github.com:LMCache/LMCache.git
cd LMCache || return
git checkout v0.1.4-alpha
cd ..
git clone git@github.com:LMCache/lmcache-server.git
cd lmcache-server || return
git checkout v0.1.1-alpha
cd ..
PYTHONPATH="$(pwd)/LMCache:$(pwd)/lmcache-vllm-extended:$PYTHONPATH"
export PYTHONPATH
python3 -m venv venv
. venv/bin/activate
pip install -r lmcache-vllm-extended/requirements.txt