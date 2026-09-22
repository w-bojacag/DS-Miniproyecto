FROM python:3.12-slim

# Crear usuario que ejecuta el tablero
RUN adduser --disabled-password --gecos '' app-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/credit-risk-dashboard

# Instalar dependencias
ADD ./credit-risk-dashboard /opt/credit-risk-dashboard/
RUN pip install --upgrade pip
RUN pip install -r /opt/credit-risk-dashboard/requirements.txt

# Hacer el directorio de trabajo ejecutable
RUN chmod +x /opt/credit-risk-dashboard/run.sh
# Cambiar propiedad de la carpeta a app-user
RUN chown -R app-user:app-user ./

USER app-user
# Puerto a exponer para el tablero
EXPOSE 8501

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
