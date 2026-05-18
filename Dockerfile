FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and frontend
COPY . .

# The API key should be passed at runtime, not baked into the image.
# Use: docker run -e KEY=sk-... -p 8000:8000 research-terminal
ENV KEY=""

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
