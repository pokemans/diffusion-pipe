#!/bin/bash
# Simple script to run the web interface

cd "$(dirname "$0")/.."
python web/app.py
