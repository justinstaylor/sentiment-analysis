FROM python:3.9-slim

RUN adduser --disabled-password appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

EXPOSE 5000

CMD ["python", "app.py"]
