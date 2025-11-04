# base image with UV + python3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# copy project files first for layer caching
COPY pyproject.toml uv.lock ./

# install deps only (no source code yet)
RUN uv sync --frozen --no-dev

# now copy source code
COPY app ./app

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app.main:combined_app", "--host", "0.0.0.0", "--port", "8080"]
