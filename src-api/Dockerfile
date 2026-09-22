FROM python:3.12-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo 
WORKDIR /opt/credit-risk-api

# Instalar dependencias
ADD ./credit-risk-api /opt/credit-risk-api/
RUN pip install --upgrade pip
RUN pip install -r /opt/credit-risk-api/requirements.txt

# Hacer el directorio de trabajo ejecutable 
RUN chmod +x /opt/credit-risk-api/run.sh
# Cambiar propiedad de la carpeta a api-user 
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la api 
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor 
CMD ["bash", "./run.sh"]
