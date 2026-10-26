#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
curl -L --fail --silent --show-error -o /tmp/nifty50.zip \
  'https://www.kaggle.com/api/v1/datasets/download/rohanrao/nifty50-stock-market-data'
unzip -oq /tmp/nifty50.zip -d data/raw
echo 'Downloaded Kaggle NIFTY-50 CSVs into data/raw.'
