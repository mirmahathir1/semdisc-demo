# SemDisc backend (Flask). Source is bind-mounted at runtime via docker-compose,
# so this image only needs the Python dependencies installed.
FROM python:3.11-slim

# System libs commonly needed by pandas / sentence-transformers wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Force Flask's polling stat-reloader (inotify is unreliable over bind mounts
    # on network filesystems), so source edits are detected and hot-reloaded.
    FLASK_DEBUG=1

WORKDIR /app/backend

# Install deps first for better layer caching. The whole repo is bind-mounted
# over /app at runtime, but the installed site-packages live outside /app.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

EXPOSE 5000

# app.py calls app.run(host="0.0.0.0", port=5000, debug=True) -> reloader on.
CMD ["python", "app.py"]
