"""Genera la cartera agregada que usa el tablero para sus vistas descriptivas.

En lugar de copiar los 150.000 registros al contenedor, se guarda el número
de clientes e incumplidos por cada combinación de segmentos. Con eso el
tablero calcula tasas de incumplimiento para cualquier combinación de filtros
sin exponer registros individuales.

Uso (desde la raíz del repo, tras src/preprocessing.py):
    python src/build_cartera.py
"""
from pathlib import Path

import pandas as pd

ENTRADA = Path("data/processed/data_prepared.csv")
SALIDA = Path("src-dash/credit-risk-dashboard/app/data/cartera_agregada.json")


def main() -> None:
    df = pd.read_csv(ENTRADA)

    df["grupo_edad"] = pd.cut(
        df["edad"], [0, 30, 40, 50, 60, 70, 200],
        labels=["18-30", "31-40", "41-50", "51-60", "61-70", "70+"])
    df["tramo_utilizacion"] = pd.cut(
        df["utilizacion"], [-0.01, 0.1, 0.25, 0.5, 0.75, 1.0, float("inf")],
        labels=["0-10%", "10-25%", "25-50%", "50-75%", "75-100%", ">100%"])
    quintiles = pd.qcut(df["ingreso_mensual"], 5,
                        labels=["Q1 (más bajo)", "Q2", "Q3", "Q4", "Q5 (más alto)"])
    df["quintil_ingreso"] = quintiles.astype(str).where(df["ingreso_mensual"].notna(),
                                                        "No reportado")
    df["moras_90_previas"] = df["moras_90"].clip(upper=3).map(
        {0: "0", 1: "1", 2: "2", 3: "3 o más"})

    dims = ["grupo_edad", "tramo_utilizacion", "quintil_ingreso", "moras_90_previas"]
    agregada = (df.groupby(dims, observed=True)["incumplio"]
                  .agg(clientes="size", incumplidos="sum")
                  .reset_index())
    for d in dims:
        agregada[d] = agregada[d].astype(str)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    agregada.to_json(SALIDA, orient="records", force_ascii=False, indent=0)
    print(f"{len(agregada)} segmentos, {agregada.clientes.sum():,} clientes -> {SALIDA}")


if __name__ == "__main__":
    main()
