FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Image already has pwuser with UID 1000 — HF requires non-root, this satisfies it
USER pwuser
ENV HOME=/home/pwuser PATH=/home/pwuser/.local/bin:$PATH
WORKDIR $HOME/app

# Python deps
COPY --chown=pwuser:pwuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# App code
COPY --chown=pwuser:pwuser . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
