# Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y gcc libpq-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
