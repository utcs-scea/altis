#!/usr/bin/env bash
set -e
set -x

GPU_NUM=${GPU_NUM:-1}

# Calling syntax: ./script app1 arguments arguments -- app2 arguments
# apps are separated by --
APPS=("$@")
if [[ "0" == "${#APPS[@]}" ]]; then
    echo "No apps provided. Please call $0 app1 args -- app2 args"
    exit
fi
value="--"
for i in "${!APPS[@]}"; do
   [[ "${APPS[$i]}" = "${value}" ]] && break
done
if [[ "$i" == "$((${#APPS[@]} - 1))" ]]; then
    # no dash separator, assume we want to run the same app on both partitions
    APP1=("${APPS[@]}")
    APP2=("${APPS[@]}")
else
    APP1=("${APPS[@]:0:$i}")
    APP2=("${APPS[@]:$((i + 1))}")
fi

clean_up_mig() {
    gpu=$1
    sudo nvidia-smi mig -i "${gpu}" -dci || true
    sudo nvidia-smi mig -i "${gpu}" -dgi || true
}

first_ci_uuid() {
    gpu=$1
    nvidia-smi -L | grep MIG | cut -d ':' -f 3 | cut -d ')' -f 1 | tr -d ' ' | head -n1
}

last_ci_uuid() {
    gpu=$1
    nvidia-smi -L | grep MIG | cut -d ':' -f 3 | cut -d ')' -f 1 | tr -d ' ' | tail -n1
}

run_cuda() {
    gpu="$1"
    shift
    CUDA_VISIBLE_DEVICES="$1"
    shift
    export CUDA_VISIBLE_DEVICES
    "$@" &
}

run_gi_ci_app_concurrent() {
    gpu=$1
    shift
    gi=$1
    shift
    ci=$1

    sudo nvidia-smi mig -i "${gpu}" -cgi "${gi}"
    sudo nvidia-smi mig -i "${gpu}" -cgi "${gi}"
    sudo nvidia-smi mig -i "${gpu}" -cci "${ci}"
    first_ci=$(first_ci_uuid "${gpu}")
    last_ci=$(last_ci_uuid "${gpu}")
    run_cuda "${gpu}" "${first_ci}" "${APP1[@]}"
    run_cuda "${gpu}" "${last_ci}" "${APP2[@]}"
    wait
    clean_up_mig "${gpu}"
}

run_ci_app_concurrent() {
    gpu=$1
    shift
    gi=$1
    shift
    ci=$1

    sudo nvidia-smi mig -i "${gpu}" -cgi "${gi}"
    sudo nvidia-smi mig -i "${gpu}" -cci "${ci}"
    sudo nvidia-smi mig -i "${gpu}" -cci "${ci}"
    first_ci=$(first_ci_uuid "${gpu}")
    last_ci=$(last_ci_uuid "${gpu}")
    run_cuda "${gpu}" "${first_ci}" "${APP1[@]}"
    run_cuda "${gpu}" "${last_ci}" "${APP2[@]}"
    wait
    clean_up_mig "${gpu}"
}

run_app_concurrent() {
    gpu=$1
    shift
    gi=$1
    shift
    ci=$1

    sudo nvidia-smi mig -i "${gpu}" -cgi "${gi}"
    sudo nvidia-smi mig -i "${gpu}" -cci "${ci}"
    first_ci="$(first_ci_uuid "${gpu}")"
    last_ci="$(last_ci_uuid "${gpu}")"
    run_cuda "${gpu}" "${first_ci}" "${APP1[@]}"
    run_cuda "${gpu}" "${last_ci}" "${APP2[@]}"
    wait
    clean_up_mig "${gpu}"
}

run_app() {
    gpu=$1
    shift
    gi=$1
    shift
    ci=$1
    shift
    app=("$@")

    sudo nvidia-smi mig -i "${gpu}" -cgi "${gi}"
    sudo nvidia-smi mig -i "${gpu}" -cci "${ci}"
    first_ci="$(first_ci_uuid)"
    run_cuda "${gpu}" "${first_ci}" "${app[@]}"
    wait
    clean_up_mig "${gpu}"
}

clean_up_mig "${GPU_NUM}"
# run in separate GIs
run_gi_ci_app_concurrent "${GPU_NUM}"  9 2
run_gi_ci_app_concurrent "${GPU_NUM}" 14 1
run_gi_ci_app_concurrent "${GPU_NUM}" 19 0

# run in same GI, separate CIs
run_ci_app_concurrent "${GPU_NUM}" 0 2
run_ci_app_concurrent "${GPU_NUM}" 5 1
run_ci_app_concurrent "${GPU_NUM}" 14 0

# run in same GI, same CI
run_app_concurrent "${GPU_NUM}" 0 4
run_app_concurrent "${GPU_NUM}" 5 3
run_app_concurrent "${GPU_NUM}" 9 2
run_app_concurrent "${GPU_NUM}" 14 1
run_app_concurrent "${GPU_NUM}" 19 0

# run app1 solo same GI, same CI
run_app "${GPU_NUM}" 0 4 "${APP1[@]}"
run_app "${GPU_NUM}" 5 3 "${APP1[@]}"
run_app "${GPU_NUM}" 9 2 "${APP1[@]}"
run_app "${GPU_NUM}" 14 1 "${APP1[@]}"
run_app "${GPU_NUM}" 19 0 "${APP1[@]}"
# run app2 solo same GI, same CI
run_app "${GPU_NUM}" 0 4 "${APP2[@]}"
run_app "${GPU_NUM}" 5 3 "${APP2[@]}"
run_app "${GPU_NUM}" 9 2 "${APP2[@]}"
run_app "${GPU_NUM}" 14 1 "${APP2[@]}"
run_app "${GPU_NUM}" 19 0 "${APP2[@]}"


