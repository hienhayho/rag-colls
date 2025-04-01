ARG PYTHON_VERSION
ARG OPENAI_API_KEY

FROM python:${PYTHON_VERSION}


RUN apt-get update && apt-get install -y curl gcc build-essential && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s $HOME/.cargo/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN uv pip install -r <(uv pip compile pyproject.toml --all-extras) && uv pip install ".[dev]"

CMD ["uv", "run", "pytest"]
