# 베이스 이미지로 python3.6-slim 사용
FROM krmp-d2hub-idock.9rum.cc/goorm/python:3.6-slim

# 빌드에 필요한 시스템 패키지 설치 (예: gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# .dockerignore에 지정한 파일 외의 모든 파일을 컨테이너로 복사
COPY . /app

RUN pip install --no-cache-dir fastapi torch

EXPOSE 8000

# 컨테이너 실행 시 FastAPI 앱을 uvicorn으로 실행
CMD ["python", "/app/app.py" ]
