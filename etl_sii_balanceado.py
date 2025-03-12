from datetime import datetime
import pandas as pd

# Ruta al archivo .xlsb
xlsx_file_path = 'data/Series_prueba.xlsx'
sheet_name = 'Hoja1'
output_csv_path = 'data/sii_total_balanceado.csv'

def excel_to_datetime(value):
    try:
        return datetime.fromtimestamp((float(value) - 25569) * 86400).strftime('%Y-%m-%d')
    except ValueError:
        return None

df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=None, skiprows=7)

columns_to_extract = [0,1]
selected_data = df[columns_to_extract]
selected_data.columns = ['date', 'total']
selected_data.to_csv(output_csv_path, index=False)


# Week as unit
df_semanal = selected_data.copy()
df_semanal.reset_index()
df_semanal['week_group'] = (df_semanal.index // 7)
df_semanal = df_semanal.groupby('week_group').agg({
    'date': 'first',  
    'total': 'sum'    
}).reset_index(drop=True)
df_semanal.to_csv("data/sii_total_semanal.csv", index=False)