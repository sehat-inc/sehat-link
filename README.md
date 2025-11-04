# sehat-link
An AI powered healthcare system

## Run Project 
```
# Activate your uv environment
source .venv/bin/activate
uv sync # Adds all dependencies

# To run entire Project
make all

# To run frontend only
make frontend

# To run backend only
make server
```

## Things You Shouldn't Touch in This Code

1. The [`combined_app`](https://github.com/sehat-inc/sehat-link/blob/e603e06fd00ce0f335002502d25e7f6b6942b6c8/app/main.py#combined_app) in `main.py`. 
It would break everything.

2. The [URL client](https://github.com/sehat-inc/sehat-link/blob/e603e06fd00ce0f335002502d25e7f6b6942b6c8/app/api/mcp/client.py#L4) listens to in `client.py`

3. The [Port set in Dockerfile](https://github.com/sehat-inc/sehat-link/blob/e603e06fd00ce0f335002502d25e7f6b6942b6c8/Dockerfile#L17) since GCP listens to 8080 by default

4. When [FastMCP is made into a ASGI app](https://github.com/sehat-inc/sehat-link/blob/e603e06fd00ce0f335002502d25e7f6b6942b6c8/app/api/mcp/server.py#L33) so that it can mount with FastAPI. 
