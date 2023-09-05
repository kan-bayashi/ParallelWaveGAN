# cuda related
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# path related
export PRJ_ROOT="${PWD}/../../.."
if [ -e "${PRJ_ROOT}/tools/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tools/venv/bin/activate"
fi

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg

# check installation
if ! command -v parallel-wavegan-train > /dev/null; then
    echo "Error: It seems setup is not finished." >&2
    echo "Error: Please setup your environment by following README.md" >&2
    return 1
fi
if ! command -v jq > /dev/null; then
    echo "Error: It seems jq is not installed." >&2
    echo "Error: Please install via \`sudo apt-get install jq\`." >&2
    echo "Error: If you do not have sudo, please download from https://stedolan.github.io/jq/download/." >&2
    return 1
fi
if ! command -v yq > /dev/null; then
    echo "Error: It seems yq is not installed." >&2
    echo "Error: Please install via \`pip install yq\`." >&2
    return 1
fi
