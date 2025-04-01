Installation
================

You can follow one of these methods:

1. **Install from PyPI**: The easiest way to install the latest stable version of `rag-colls` is via `pip`:

   .. code-block:: shell

        pip install rag-colls

2. **Install from source**: If you want to install the latest development version, you can clone the repository and install it manually:

    .. code-block:: shell

        # Clone the repository
        git clone https://github.com/hienhayho/rag-colls.git
        cd rag-colls/

        pip install -v .

3. **Install with docker**: If you prefer to use Docker, you can build the image from the Dockerfile provided in the repository:

    .. code-block:: shell

       # Clone the repository
        git clone https://github.com/hienhayho/rag-colls.git
        cd rag-colls/

        # Docker build
        DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
                            --build-arg OPENAI_API_KEY=<YOUR_OPENAI_KEY> \
                            --build-arg PYTHON_VERSION="3.10" \
                            -t rag-colls:3.10 .
