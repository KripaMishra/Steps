FROM python:3.11-slim

ARG REQUIREMENTS=requirements-demo.txt
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r "$REQUIREMENTS" \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

ENTRYPOINT ["streamlit", "run", "main.py", "--server.headless=true"]
