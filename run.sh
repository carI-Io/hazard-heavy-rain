#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ccpy4
cd /home/admin_climatecharted_com/GitHub/hazard-heavy-rain

python more_idf_italy.py \
    --data-root /mnt/data/more \
    --output-dir /home/admin_climatecharted_com/data/MOloch/IDF_results \
    --years 1991-2020 \
    --log-file ./more_idf.log

python more_chicago_hyetograph.py \
    --output-dir /home/admin_climatecharted_com/data/MOloch/IDF_results \
    --target-rp 100 \
    --chicago-r 0.35 \
    --chicago-dt 5 \
    --log-file ./chicago.log \
    --skip-inspect \
    --skip-plots

git add .
git commit -a -m "autocommit: update IDF" || true
git pull
git push

gsutil -m cp -r /home/admin_climatecharted_com/data/MOloch/IDF_results \
    gs://cc-geodata-bucket/