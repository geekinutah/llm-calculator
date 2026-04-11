FROM python:3.12-slim

LABEL org.opencontainers.image.authors="Mike Wilson <geekinutah@gmail.com>"
LABEL description="LLM GPU Throughput Calculator - zero-dependency static server"

# Create non-root user
RUN addgroup --system --gid 1001 appgroup && \
    adduser  --system --uid 1001 --ingroup appgroup appuser

WORKDIR /app

# Copy application files
COPY server.py    ./
COPY static/      ./static/

# Set ownership
RUN chown -R appuser:appgroup /app

USER appuser

ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" || exit 1

CMD ["python3", "server.py"]
