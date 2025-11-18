FROM python:3.11-slim

WORKDIR /app

# Dependencies untuk OpenCV, dll (sesuaikan kalau error)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./


RUN python -m pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

COPY src ./src
COPY settings.yaml ./settings.yaml

RUN mkdir -p artifacts outputs

ENTRYPOINT ["bsort"]
