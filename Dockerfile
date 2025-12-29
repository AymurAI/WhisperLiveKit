# Build stage
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy


WORKDIR /app

ARG EXTRAS
ARG HF_PRECACHE_DIR
ARG HF_TKN_FILE

# Install build dependencies and Python
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  build-essential \
  ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install PyTorch with CUDA support first (before other dependencies)
RUN --mount=type=tmpfs,target=/root/.cache/uv \
  uv venv -p 3.12 && \
  uv pip install --index-url https://download.pytorch.org/whl/cu129 torch torchaudio

# Copy only necessary files for dependency resolution
COPY pyproject.toml uv.lock ./

# Install dependencies without the project itself (for better caching)
RUN --mount=type=tmpfs,target=/root/.cache/uv \
  uv sync --frozen --no-install-project

# Copy the source code
COPY whisperlivekit ./whisperlivekit

# Sync the project with optional extras
RUN --mount=type=tmpfs,target=/root/.cache/uv \
  if [ -n "$EXTRAS" ]; then \
  echo "Installing with extras: $EXTRAS"; \
  uv sync --frozen --no-editable --extra "$EXTRAS"; \
  else \
  echo "Installing base package only"; \
  uv sync --frozen --no-editable; \
  fi

# Copy pre-cached models if provided
RUN mkdir -p /root/.cache/huggingface
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
  echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
  mkdir -p /root/.cache/huggingface/hub && \
  cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
  else \
  echo "No local Hugging Face cache specified, skipping copy"; \
  fi

# Copy Hugging Face token if provided
RUN if [ -n "$HF_TKN_FILE" ]; then \
  echo "Copying Hugging Face token from $HF_TKN_FILE"; \
  mkdir -p /root/.cache/huggingface && \
  cp $HF_TKN_FILE /root/.cache/huggingface/token; \
  else \
  echo "No Hugging Face token file specified, skipping token setup"; \
  fi

# Runtime stage
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  ffmpeg \
  ca-certificates && \
  rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.local /root/.local
ENV PATH="/app/.venv/bin:$PATH"



# Copy Hugging Face cache from builder (if exists)
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Make the cache directory persistent via volume
VOLUME ["/root/.cache/huggingface/hub"]

EXPOSE 8000

ENTRYPOINT ["uv", "run", "whisperlivekit-server", "--host", "0.0.0.0"]

CMD ["--model", "medium", "--diarization", "--pcm-input"]
