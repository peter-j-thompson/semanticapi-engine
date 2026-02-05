FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0
ENV PROVIDERS_DIR=/app/providers

CMD ["uvicorn", "semanticapi.server:app", "--host", "0.0.0.0", "--port", "8080"]
