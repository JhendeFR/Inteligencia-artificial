import os
import pandas as pd
from glob import glob

DATA_FOLDER = r"C:\Users\jhean\OneDrive\Documentos\Tareas\Inteligencia artificial\1erParcial\Dataset"
OUTPUT_FILE = r"C:\Users\jhean\OneDrive\Documentos\Tareas\Inteligencia artificial\1erParcial\EIA923_Cleaned_2020_2024.xlsx"

# Funcion para extraer datos limpios desde cada archivo
def extract_generation_data(filepath):
    try:
        # Cargar desde la fila 6
        df = pd.read_excel(filepath, sheet_name='Page 1 Generation and Fuel Data', skiprows=5)
        df = df.dropna(axis=1, how='all') # Eliminar columnas vacías
        if "Plant Id" not in df.columns:
            return None

        df = df[df["Plant Id"].notna()] # Eliminar filas sin ID de planta

        # Seleccionar columnas clave (que creo que son necesarias)
        col_map = {
            "Plant Id": "Plant_ID",
            "Plant Name": "Plant_Name",
            "Plant State": "State",
            "Sector Number": "Sector",
            "Energy Source": "Fuel_Type",
            "Fuel Consumption\nQuantity": "Fuel_Consumed_Quantity",
            "Total Fuel Consumption\nMMBtu": "Fuel_Consumed_MMBtu",
            "Net Generation\n(Megawatthours)": "Net_Generation_MWh",
            "YEAR": "Year"
        }

        # Filtrar columnas disponibles y renombrar
        columns_existing = {k: v for k, v in col_map.items() if k in df.columns}
        df = df[list(columns_existing.keys())].rename(columns=columns_existing)

        return df

    except Exception as e:
        print(f"[ERROR] No se pudo procesar: {filepath}\n{e}")
        return None

def main():
    file_pattern = os.path.join(DATA_FOLDER, "EIA923_Schedules_2_3_4_5_M_12_20*.xlsx")
    excel_files = sorted(glob(file_pattern))

    print(f"Se encontraron {len(excel_files)} archivos.")

    # Extraer y combinar
    all_data = []
    for file in excel_files:
        print(f"Procesando: {file}")
        df = extract_generation_data(file)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No se pudieron procesar archivos.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_excel(OUTPUT_FILE, index=False)
    print(f" Archivo combinado guardado como: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    #Con esto tenemos 70.906 filas y 6 columnas con datos limpios y razonables para entrenar los modelos
    #
