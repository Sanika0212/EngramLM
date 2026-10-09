# CUDA image for running EngramLM experiments on a GPU machine.
#   docker build -t engramlm .
#   docker run --gpus all -v $PWD/runs:/app/runs engramlm lm --config configs/lm-smollm2.yaml --out runs/lm --device cuda
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf-cache

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[train]"

ENTRYPOINT ["engramlm"]
CMD ["--help"]
