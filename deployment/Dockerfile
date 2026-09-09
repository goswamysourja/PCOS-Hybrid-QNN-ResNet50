
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best_model.pth .

EXPOSE 7860

CMD ["python", "app.py"]

