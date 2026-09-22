#!/bin/bash

# Puerto del tablero (por defecto 8501). Se puede definir con -e PORT o --env-file
PORT=${PORT:-8501}

exec streamlit run app/tablero.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
