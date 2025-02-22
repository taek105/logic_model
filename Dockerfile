FROM python:3.6-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn torch numpy

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
