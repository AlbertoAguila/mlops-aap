FROM python:3.11-slim

WORKDIR /app

# Copiar e instalar dependencias primero (aprovecha caché de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

EXPOSE 7860

# Placeholder — Compose lo sobreescribe con http://backend:8000
ENV BACKEND_URL=http://localhost:8000

CMD ["python", "gradio_app.py"]
