# ── Base image ─────────────────────────────────────────────────────────────────
# python:3.11-slim is the standard production Python base.
# "slim" strips dev tools and docs — smaller image, faster pull.
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential: needed to compile some Python packages (e.g. chromadb)
# curl: used in the HEALTHCHECK below to ping Streamlit
# No cleanup needed — Docker layer caching means this only runs when changed.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
# All subsequent commands run from here inside the container.
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
# Copy requirements BEFORE copying source code.
# Why? Docker caches layers. If only source code changes (not requirements),
# this layer is reused from cache — no reinstalling packages on every build.
# This is the most important Docker optimisation for Python projects.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
# Copied after pip install so code changes don't bust the pip cache.
COPY . .

# ── Environment ────────────────────────────────────────────────────────────────
# GROQ_API_KEY is NOT baked into the image — it's injected at runtime
# via docker-compose.yml (from your .env file). Never hardcode secrets.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ───────────────────────────────────────────────────────────────
# Docker polls this every 30s. If it fails 3 times, the container is marked
# unhealthy. Useful for orchestration (Kubernetes, ECS) and monitoring.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# --server.address=0.0.0.0 makes Streamlit bind to all interfaces,
# not just localhost — required for the port to be accessible outside
# the container. Without this, the app runs but nothing can reach it.
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]