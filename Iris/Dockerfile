# Usar imagen base ligera de Python
FROM python:3.11-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código (el .dockerignore evitará que se copie basura)
COPY . .

# Exponer el puerto 8000
EXPOSE 8000

# Comando para correr la app (usando tu script que ya tiene uvicorn dentro)
CMD ["python", "src/app.py"]