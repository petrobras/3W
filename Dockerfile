# The builder image, used to build the virtual environment
FROM python:3.10.11-bullseye AS builder

RUN apt-get update && apt-get install -y git curl \
    texlive-latex-recommended texlive-latex-extra texlive-pictures texlive-fonts-recommended texlive-latex-extra-doc
    
RUN pip install poetry==2.1.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --with dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.10.11-slim-bullseye AS runtime

RUN apt-get update && apt-get install -y git curl bash && apt-get clean

RUN apt-get install \
        build-essential \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-cffi \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        shared-mime-info


ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY . /app

WORKDIR /app

ENTRYPOINT ["/bin/bash"]
