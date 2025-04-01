ARG PYTHON_VERSION
ARG OPENAI_API_KEY

FROM python:${PYTHON_VERSION}


RUN apt-get update && apt-get install -y curl gcc build-essential && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN pip install -v . && pip install ".[dev]"

CMD ["pytest"]
