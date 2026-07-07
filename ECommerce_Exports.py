from __future__ import annotations
from pathlib import Path
import pandas as pd

#Función que saca timezones
def _remove_timezone_from_series(series):
    if pd.api.types.is_datetime64tz_dtype(series):
        return series.dt.tz_localize(None)

    if series.dtype == "object":
        return series.apply(
            lambda x: x.tz_localize(None)
            if isinstance(x, pd.Timestamp) and x.tz is not None
            else x
        )

    return series

#Función que saca timezones de los dataframes
def _remove_timezone_from_dataframe(df):
    df = df.copy()
    for col in df.columns:
        df[col] = _remove_timezone_from_series(df[col])
    return df

#Función de exportar
def ExportarDataFramesExcel(dataframes, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output) as writer:
        for sheet_name, df in dataframes.items():
            if df is None:
                continue

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            df = _remove_timezone_from_dataframe(df)
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return output