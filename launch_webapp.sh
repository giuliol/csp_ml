#!/usr/bin/env bash

cwd=$(pwd)
(cd src/web && PYTHONPATH=$cwd  python3.4 webapp.py)