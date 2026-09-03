import pandas as pd

def preprocess_data(input_path, output_path):
    """
    Lee datos crudos, aplica reglas de limpieza basadas en 01_eda.pynb 
    y guarda los datos procesados.
    """
    print(f"Cargando datos desde {input_path}...")
    df = pd.read_csv(input_path, index_col=0).copy()
    
    # Renombrar columnas para facilitar el manejo
    df = df.rename(columns={
        "SeriousDlqin2yrs": "incumplio",
        "RevolvingUtilizationOfUnsecuredLines": "utilizacion",
        "age": "edad",
        "NumberOfTime30-59DaysPastDueNotWorse": "moras_30_59",
        "DebtRatio": "razon_deuda",
        "MonthlyIncome": "ingreso_mensual",
        "NumberOfOpenCreditLinesAndLoans": "lineas_abiertas",
        "NumberOfTimes90DaysLate": "moras_90",
        "NumberRealEstateLoansOrLines": "creditos_inmob",
        "NumberOfTime60-89DaysPastDueNotWorse": "moras_60_89",
        "NumberOfDependents": "dependientes",
    })
    
    # 1. Filtrar registros con edad <= 0
    df = df[df['edad'] > 0]
    
    # 2. Variable indicadora de ingreso faltante e imputación
    df['ingreso_faltante'] = df['ingreso_mensual'].isna().astype(int)
    df['ingreso_mensual'] = df['ingreso_mensual'].fillna(df['ingreso_mensual'].median())
    
    # 3. Imputar dependientes
    df['dependientes'] = df['dependientes'].fillna(df['dependientes'].median())
    
    # 4. Tratamiento de centinelas (moras 96/98)
    cols_moras = ['moras_30_59', 'moras_60_89', 'moras_90']
    df['es_moro_cronico'] = df[cols_moras].isin([96, 98]).any(axis=1).astype(int)
    df[cols_moras] = df[cols_moras].replace([96, 98], 0)
    
    # 5. Winsorización
    limite_superior = df['utilizacion'].quantile(0.99)
    df['utilizacion'] = df['utilizacion'].clip(upper=limite_superior)
    
    print(f"Guardando datos procesados en {output_path}...")
    df.to_csv(output_path, index=True)
    print("Preprocesamiento completado exitosamente.")

if __name__ == "__main__":
    # invocamos la función limoieza de datos
    preprocess_data("data/raw/cs-training.csv", "data/processed/data_clean.csv")
