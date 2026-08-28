#!/usr/bin/env bash
set -u

threshold_mib="${GPU_FREE_THRESHOLD_MIB:-13500}"
run_id="${1:-unified_organs_v4_mad5_retry_2018_2022_20260826}"
project="/home/yym/min_gp/分钟级因子挖掘"

while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
    printf '[%s] waiting for GPU: free_mib=%s threshold_mib=%s\n' \
        "$(date '+%F %T')" "$free_mib" "$threshold_mib"
    if [ "$free_mib" -ge "$threshold_mib" ]; then
        cd "$project"
        exec env \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            MIN_GP_DATA_ROOT=/home/yym/min_gp_data \
            /home/yym/min_gp/.venv/bin/python -u -m seed_tree_gp \
            --start 2018-01-02 \
            --end 2022-12-31 \
            --pop 60 \
            --gens 8 \
            --max-depth 5 \
            --max-peak-bytes 12000000000 \
            --outlier-mad 5 \
            --run-id "$run_id"
    fi
    sleep 30
done
