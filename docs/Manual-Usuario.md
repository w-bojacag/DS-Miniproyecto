# SISTEMA DE ANÁLISIS Y PREDICCIÓN DE RIESGO CREDITICIO

## MANUAL DE USUARIO

Este manual explica cómo usar el tablero para evaluar el riesgo de incumplimiento de un solicitante de crédito y cómo interpretar el resultado. Para instalar o desplegar el sistema, ver el [Manual de Instalación](Manual-Instalacion.md).

## 1. ¿Qué hace el sistema?

El usuario ingresa los datos financieros y de historial crediticio de un solicitante en el tablero. El tablero envía esos datos a la API de predicción, que evalúa un modelo de machine learning (XGBoost) entrenado sobre el dataset *Give Me Some Credit* y devuelve:

- La probabilidad de que el solicitante incumpla (mora grave) en los próximos 2 años.
- Una clasificación de riesgo (**Riesgo Alto** / **Riesgo Bajo**).
- Una explicación de las variables que más influyeron en esa predicción.

El resultado es un apoyo a la decisión de aprobación de crédito, no un reemplazo del criterio del analista.

## 2. Acceder al tablero

Abra en el navegador la URL del tablero que le indique quien lo desplegó, por ejemplo:

- Despliegue local: `http://localhost:8501`
- Despliegue en servidor: `http://<IP-o-dominio-del-tablero>:8501`

Si la página no carga, verifique con la persona responsable del despliegue que los contenedores `credit-dash` y `credit-api` estén corriendo (ver Manual de Instalación, sección 5.C).

## 3. Completar el formulario

El formulario está dividido en dos columnas.

### 3.1 Datos demográficos y financieros

| Campo | Descripción |
|---|---|
| Edad del solicitante | Edad en años. |
| Ingreso Mensual ($) | Ingreso mensual del solicitante. Puede dejarse vacío si el solicitante no lo reporta; el modelo lo trata como un dato faltante. |
| Número de dependientes | Personas que dependen económicamente del solicitante. |
| Razón de Deuda (Obligaciones / Ingreso) | Relación entre las obligaciones mensuales del solicitante y su ingreso. |

### 3.2 Comportamiento e historial

| Campo | Descripción |
|---|---|
| Utilización del límite de crédito (0 a 1) | Proporción del cupo de tarjetas/líneas de crédito que el solicitante tiene actualmente utilizado. |
| Líneas de crédito abiertas | Número de créditos o tarjetas abiertas a nombre del solicitante. |
| Créditos hipotecarios | Número de créditos o líneas hipotecarias que tiene el solicitante. |
| Moras de 30-59 días | Veces que el solicitante se ha atrasado entre 30 y 59 días en un pago. |
| Moras de 60-89 días | Veces que se ha atrasado entre 60 y 89 días. |
| Moras de 90+ días | Veces que se ha atrasado 90 días o más. |

Todos los campos numéricos tienen un valor por defecto razonable; ajústelos según los datos reales del solicitante que se está evaluando.

## 4. Evaluar al solicitante

Al presionar **Evaluar Solicitante**, el tablero envía los datos a la API y muestra el resultado debajo del formulario.

### 4.1 Resultado de la evaluación

- Una barra de progreso con la **probabilidad de incumplimiento** (0% a 100%).
- Un mensaje de color según la clasificación:
  - 🔴 **RIESGO ALTO**: la probabilidad supera el umbral de decisión del modelo. Recomendación mostrada: *Revisión Manual / Rechazar*.
  - 🟢 **RIESGO BAJO**: la probabilidad está por debajo del umbral. Recomendación mostrada: *Aprobar*.

El umbral de decisión no es 50%: se calibró sobre los datos de entrenamiento buscando el mejor balance entre detectar incumplimientos reales y no rechazar de más a buenos pagadores. Por eso una probabilidad de, por ejemplo, 60% puede seguir clasificarse como Riesgo Bajo si está por debajo de ese umbral.

### 4.2 Explicación del modelo

Debajo del resultado aparece una tabla con las variables que más pesaron en esa predicción puntual (no en el modelo en general), con cuatro columnas:

| Columna | Significado |
|---|---|
| Variable | Nombre de la variable evaluada. |
| Valor | El valor que se ingresó para esa variable. |
| Aporte | Qué tanto empujó esa variable la predicción, en escala logarítmica (*log-odds*). |
| Dirección | 🔴 "aumenta" el riesgo de incumplimiento, o 🟢 "reduce" el riesgo. |

Se muestran las 4 variables con mayor peso (positivo o negativo) para ese solicitante específico. Dos solicitantes distintos pueden tener explicaciones con variables diferentes, según cuáles de sus datos sean atípicos.

## 5. Errores comunes

| Mensaje | Causa probable | Qué hacer |
|---|---|---|
| ❌ Error al conectar con la API | La API no está corriendo, o el tablero no puede alcanzarla en la red. | Avisar a la persona responsable del despliegue; ver Manual de Instalación, sección 3.2 y 5.C. |
| ❌ Error procesando la respuesta | La API respondió pero con un formato inesperado (por ejemplo, un error de validación). | Verificar que los valores ingresados sean numéricos y estén dentro de los rangos permitidos por el formulario. |
| El formulario no deja avanzar / valores en rojo | Un campo numérico está fuera del rango permitido (por ejemplo, edad negativa). | Ajustar el valor dentro del rango indicado en el propio campo. |

## 6. Alcance y limitaciones

- El modelo fue entrenado con datos históricos y una definición específica de incumplimiento (mora grave en 2 años); no debe usarse para decisiones fuera de ese contexto sin una nueva validación.
- La predicción es un apoyo a la decisión, no una aprobación o rechazo automático del crédito.
- El modelo no debe usarse para evaluar solicitantes con perfiles muy distintos a los del dataset de entrenamiento (por ejemplo, otro país o segmento de mercado) sin antes revalidar su desempeño.
