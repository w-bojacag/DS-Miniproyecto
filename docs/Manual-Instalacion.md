# SISTEMA DE ANÁLISIS Y PREDICCIÓN DE RIESGO CREDITICIO

## MANUAL DE INSTALACIÓN

Este manual describe cómo desplegar el sistema en contenedores Docker. El sistema tiene dos servicios independientes, cada uno con su propia carpeta, su propio Dockerfile y su propia imagen:

| Servicio | Carpeta | Imagen | Puerto |
|---|---|---|---|
| API de predicción (FastAPI) | `src-api/` | `credit-api` | 8001 |
| Tablero (Streamlit) | `src-dash/` | `credit-dash` | 8501 |

El tablero envía los datos del solicitante a la API por HTTP y muestra la predicción que esta devuelve. Los dos servicios pueden desplegarse de dos formas:

| Modalidad | Cuándo usarla | Red de Docker | Valor de `API_URL` en el tablero |
|---|---|---|---|
| **A. Mismo equipo** | Ambos contenedores corren en la misma máquina. | Necesaria (`credit-net`). | `http://credit-api` (nombre del contenedor de la API). |
| **B. Distribuido** | La API y el tablero corren en ubicaciones distintas. | No aplica. | `http://` seguido de la IP o el dominio público de la máquina de la API. |

Los pasos 1 a 4 son comunes a ambas modalidades. El paso 5 se divide según la modalidad elegida.

## 1. Requisitos

- Docker Desktop (Windows/Mac) o Docker Engine (Linux), en ejecución. En modalidad distribuida, en cada máquina que aloje un servicio.
- Git.

Verifique la instalación:

```bash
docker --version
git --version
```

## 2. Clonar el repositorio

> git clone https://github.com/w-bojacag/DS-Miniproyecto.git

Los comandos de los siguientes pasos se ejecutan desde la raíz del repositorio clonado. En modalidad distribuida, clone el repositorio en cada máquina (o solo la carpeta del servicio que alojará).

## 3. Configuración de variables de entorno (`.env`)

Cada servicio lee su configuración desde un archivo `.env` ubicado en su carpeta. Estos archivos no se versionan (están en `.gitignore`), por lo que deben crearse manualmente en cada máquina donde se despliegue el servicio.

### 3.1 API: `src-api/.env`

```env
PORT=8001
RISK_THRESHOLD=0.40
```

| Variable | Descripción |
|---|---|
| `PORT` | Puerto en el que la API escucha dentro del contenedor. **Obligatoria**: sin ella la API no inicia. |
| `RISK_THRESHOLD` | Umbral que representa una política general para convertir una probabilidad en una decisión. **Obligatoria**: sin ella la API no inicia. |

### 3.2 Tablero: `src-dash/.env`

El contenido depende de la modalidad.

**Modalidad A: mismo equipo**

```env
PORT=8501
API_URL=http://credit-api
API_PORT=8001
API_ENDPOINT=/api/v1/predict
LOG_LEVEL=info
```

**Modalidad B: distribuido**

```env
PORT=8501
API_URL=http://api.midominio.com
API_PORT=8001
API_ENDPOINT=/api/v1/predict
LOG_LEVEL=info
```

Reemplace `api.midominio.com` por el dominio o la IP pública de la máquina donde corre la API (por ejemplo, `http://203.0.113.10`).

| Variable | Descripción |
|---|---|
| `PORT` | Puerto en el que el tablero escucha dentro del contenedor. Si no se define, usa 8501. |
| `API_URL` | Dirección de la API, incluyendo el protocolo (`http://`): nombre del contenedor (modalidad A) o IP/dominio público (modalidad B). |
| `API_PORT` | Puerto de la API. En modalidad A debe coincidir con `PORT` de `src-api/.env`. En modalidad B es el puerto publicado de la API en su máquina. |
| `API_ENDPOINT` | Ruta del servicio de predicción de la API. |
| `LOG_LEVEL` | Nivel de registro del tablero. |

Tenga en cuenta al definir estas variables:

- La petición a la API la hace el contenedor del tablero (no el navegador del usuario). Por eso la API debe ser alcanzable desde la máquina del tablero.
- No use `localhost` ni `127.0.0.1` en `API_URL`: dentro del contenedor del tablero apuntan al propio tablero y no a la API.
- La API de este manual se publica por `http`. Si la expone detrás de un certificado (`https`), ajuste `API_URL` y `API_PORT` en consecuencia.

### 3.3 Relación entre `PORT` y el puerto publicado

`PORT` es el puerto **dentro del contenedor**. El puerto por el que se accede desde el equipo se define en el `docker run` con `-p <puerto-del-equipo>:<PORT>`. El valor de la derecha debe coincidir con `PORT`. Por ejemplo, con `PORT=9000` el mapeo sería `-p 8001:9000`.

## 4. Construcción de las imágenes

Construir una imagen crea únicamente la imagen, no inicia ningún contenedor. Cada imagen se construye en la máquina donde se va a ejecutar. El Dockerfile de cada servicio está en la raíz de su carpeta (`src-api/` o `src-dash/`), que es también el contexto de construcción.

### 4.1 Imagen de la API

```bash
docker build -t credit-api ./src-api
```

### 4.2 Imagen del tablero

```bash
docker build -t credit-dash ./src-dash
```

### 4.3 Verificar las imágenes

```bash
docker images
```

Deben aparecer `credit-api` y/o `credit-dash`, según lo construido en esa máquina.

## 5. Despliegue en Docker

Inicie siempre la API primero y luego el tablero.

### 5.A Modalidad A: mismo equipo

#### 5.A.1 Crear la red

Se crea una red compartida para que el tablero encuentre la API por nombre. Solo es necesario una vez.

```bash
docker network create credit-net
```

Si la red ya existe, Docker lo indicará y puede continuar. Para comprobarlo: `docker network ls`.

#### 5.A.2 Iniciar la API

```bash
docker run -d --name credit-api --network credit-net --env-file src-api/.env -p 8001:8001 credit-api
```

#### 5.A.3 Iniciar el tablero

```bash
docker run -d --name credit-dash --network credit-net --env-file src-dash/.env -p 8501:8501 credit-dash
```

El nombre `--name credit-api` debe coincidir con el host de `API_URL` en `src-dash/.env`.

### 5.B Modalidad B: distribuido

En esta modalidad no se usa la red `credit-net`: cada contenedor corre de forma independiente y se comunican a través de la red pública o corporativa.

#### 5.B.1 Máquina de la API

1. Construya la imagen (paso 4.1) y cree `src-api/.env`.
2. Inicie el contenedor:

```bash
docker run -d --name credit-api --env-file src-api/.env -p 8001:8001 credit-api
```

3. Habilite el puerto **8001/TCP** de entrada en el firewall de la máquina (y en el grupo de seguridad o reglas de red del proveedor, si aplica).
4. Compruebe desde otra máquina que la API responde: `http://<IP-o-dominio-de-la-API>:8001/api/v1/health`.

#### 5.B.2 Máquina del tablero

1. Construya la imagen (paso 4.2) y cree `src-dash/.env` con la IP o dominio de la API (paso 3.2, modalidad B).
2. Inicie el contenedor:

```bash
docker run -d --name credit-dash --env-file src-dash/.env -p 8501:8501 credit-dash
```

3. Habilite el puerto **8501/TCP** de entrada en el firewall si el tablero debe ser accesible desde otras máquinas.

### 5.C Verificar el despliegue

En cada máquina:

```bash
docker ps
```

Deben aparecer los contenedores correspondientes con estado `Up`.

| Servicio | URL (modalidad A) | URL (modalidad B) |
|---|---|---|
| Tablero | http://localhost:8501 | http://\<IP-o-dominio-del-tablero\>:8501 |
| Estado del tablero | http://localhost:8501/_stcore/health | http://\<IP-o-dominio-del-tablero\>:8501/_stcore/health |
| Documentación de la API (Swagger) | http://localhost:8001/docs | http://\<IP-o-dominio-de-la-API\>:8001/docs |
| Estado de la API | http://localhost:8001/api/v1/health | http://\<IP-o-dominio-de-la-API\>:8001/api/v1/health |

Para comprobar la integración, abra el tablero, complete el formulario y pulse **Evaluar Solicitante**: debe mostrarse el resultado de la predicción.

## 6. Operación

Ver los registros de un contenedor:

```bash
docker logs credit-api
docker logs credit-dash
```

Detener e iniciar los contenedores (en modalidad distribuida, cada comando en la máquina que aloja el servicio). Detenga primero el tablero y arranque primero la API:

```bash
docker stop credit-dash credit-api
docker start credit-api credit-dash
```

Eliminar los contenedores (por ejemplo, antes de reconstruir las imágenes) y, en modalidad A, la red:

```bash
docker rm -f credit-dash credit-api
docker network rm credit-net
```

Después de modificar un `.env`, elimine el contenedor y créelo de nuevo con `docker run`: las variables solo se leen al crearlo.
