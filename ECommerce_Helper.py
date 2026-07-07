import os
from collections import Counter
import numpy as np
import pandas as pd


def _ValidarFraccionMuestra(session_sample_frac):
    #Esta fraccion se interpreta como la probabilidad de quedarse con una sesion.
    if not 0 < session_sample_frac <= 1:
        raise ValueError("session_sample_frac debe estar entre (0, 1].")


def _ConstruirMascaraHashSesion(session_series, session_sample_frac):
    #Se usa un hash estable para que la muestra no dependa del orden del archivo.
    _ValidarFraccionMuestra(session_sample_frac)

    #Antes de hashear, se normaliza la sesion para que los nulos no rompan el proceso.
    sesiones_normalizadas = session_series.fillna("").astype("string")
    hashes_sesion = pd.util.hash_pandas_object(sesiones_normalizadas, index=False,).to_numpy(dtype=np.uint64)
    #La idea es que si el hash cae bajo cierto umbral, esa sesion entra a la muestra.
    umbral = np.uint64(session_sample_frac * np.iinfo(np.uint64).max)
    return hashes_sesion <= umbral


def PreprocesarEventos(chunk):
    #Aqui se dejan listas solo las columnas que despues usamos en el analisis.
    eventos = chunk.copy()
    eventos["event_time"] = pd.to_datetime(eventos["event_time"], utc=True, errors="coerce")
    eventos["event_type"] = eventos["event_type"].astype("string").str.lower()
    eventos["category_code"] = eventos["category_code"].fillna("unknown").astype("string")
    eventos["brand"] = eventos["brand"].fillna("unknown").astype("string")
    eventos["price"] = pd.to_numeric(eventos["price"], errors="coerce")

    #Columnas minimas para reconstruir una sesion de forma util.
    eventos = eventos.dropna(subset=["event_time", "product_id", "user_session"]).copy()
    eventos["product_id"] = eventos["product_id"].astype("int64")
    return eventos


def CargarEventosRepresentativosCsv(data_path, file_name, use_cols=None, dtypes=None, session_sample_frac=0.01, chunksize=1_000_000, verbose=True,):
    #El CSV se lee por bloques y se va filtrando.
    file_path = os.path.join(data_path, file_name)
    lector = pd.read_csv(file_path, usecols=use_cols, dtype=dtypes, chunksize=chunksize, low_memory=False,)

    bloques_muestra = []
    filas_escaneadas = 0
    filas_muestra = 0

    for chunk_idx, chunk in enumerate(lector, start=1):
        #Cada bloque se preprocesa y se filtra de inmediato para no cargar el CSV completo en memoria.
        filas_escaneadas += len(chunk)
        eventos_preparados = PreprocesarEventos(chunk)
        mascara_muestra = _ConstruirMascaraHashSesion(eventos_preparados["user_session"], session_sample_frac,)
        eventos_muestra = eventos_preparados.loc[mascara_muestra].copy()

        if eventos_muestra.empty:
            if verbose:
                print(f"Chunk {chunk_idx}: escaneadas={filas_escaneadas:,}, " "muestra acumulada=0")
            continue

        bloques_muestra.append(eventos_muestra)
        filas_muestra += len(eventos_muestra)

        if verbose:
            print(f"Chunk {chunk_idx}: escaneadas={filas_escaneadas:,}, " f"muestra acumulada={filas_muestra:,}")

    if not bloques_muestra:
        #Si nada entra en la muestra, se devuelve un DataFrame vacio pero con el formato esperado.
        vacio = pd.DataFrame(columns=list(use_cols) if use_cols is not None else None)
        return PreprocesarEventos(vacio)

    #Al final se ordena por sesion y tiempo para que la parte secuencial quede lista.
    eventos = pd.concat(bloques_muestra, ignore_index=True)
    eventos = eventos.sort_values(["user_session", "event_time"]).reset_index(drop=True)
    return eventos


def ConstruirResumenMuestraCargada(events):
    #Resumen para comprobar rapido si la muestra quedo razonable.
    if events.empty:
        return pd.DataFrame([{"metrica": "eventos_cargados", "valor": 0}, {"metrica": "sesiones_unicas", "valor": 0}, {"metrica": "productos_unicos", "valor": 0},])

    #Se guardan metricas sencillas para validar cobertura, volumen y rango temporal.
    resumen = [
        {"metrica": "eventos_cargados", "valor": int(len(events))},
        {"metrica": "sesiones_unicas", "valor": int(events["user_session"].nunique())},
        {"metrica": "usuarios_unicos", "valor": int(events["user_id"].nunique(dropna=True))},
        {"metrica": "productos_unicos", "valor": int(events["product_id"].nunique())},
        {"metrica": "fecha_min", "valor": events["event_time"].min()},
        {"metrica": "fecha_max", "valor": events["event_time"].max()},
        {"metrica": "dias_cubiertos", "valor": int(events["event_time"].dt.normalize().nunique())},
    ]
    return pd.DataFrame(resumen)


def ConstruirDistribucionDiaria(events):
    #Se ve si la muestra quedo repartida a lo largo del mes o muy concentrada.
    if events.empty:
        return pd.DataFrame(columns=["date", "event_count"])

    #Se baja la fecha a nivel dia para no depender de horas o minutos al inspeccionar cobertura.
    distribucion = (events.assign(date=events["event_time"].dt.date.astype("string")).groupby("date").size().rename("event_count").reset_index().sort_values("date").reset_index(drop=True))
    return distribucion


def SepararSesionesPorVentanasDeTiempo(session_df, validation_days=3, test_days=3,):
    #Se seaparan los ultimos dias para validacion y prueba manteniendo el orden del tiempo.
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days y test_days deben ser enteros positivos.")

    sesiones_ordenadas = session_df.sort_values("end_time").reset_index(drop=True)
    fechas_fin = sesiones_ordenadas["end_time"].dt.normalize()
    fechas_unicas = fechas_fin.drop_duplicates().sort_values().reset_index(drop=True)

    dias_necesarios = validation_days + test_days + 1
    if len(fechas_unicas) < dias_necesarios:
        raise ValueError("No hay suficientes fechas distintas para crear ventanas de entrenamiento, validacion y prueba.")

    #Las ventanas se arman desde el final del periodo.
    inicio_test = fechas_unicas.iloc[-test_days]
    inicio_validacion = fechas_unicas.iloc[-(test_days + validation_days)]

    train_df = sesiones_ordenadas[fechas_fin < inicio_validacion].copy()
    val_df = sesiones_ordenadas[(fechas_fin >= inicio_validacion) & (fechas_fin < inicio_test)].copy()
    test_df = sesiones_ordenadas[fechas_fin >= inicio_test].copy()

    #A partir de este punto, validation y test solo pueden usar items que existan en train.
    items_train = set(item for secuencia in train_df["items"] for item in secuencia)

    def MantenerItemsConocidos(secuencia):
        #En validacion y test no deberian aparecer items que train nunca vio.
        return [item for item in secuencia if item in items_train]

    for split_df in (val_df, test_df):
        split_df["items"] = split_df["items"].apply(MantenerItemsConocidos)
        #La longitud puede cambiar mucho despues de quitar items desconocidos.
        split_df["session_len"] = split_df["items"].apply(len)

    #Si una sesion queda con menos de 2 items, ya no sirve para next-item prediction.
    val_df = val_df[val_df["session_len"] >= 2].copy().reset_index(drop=True)
    test_df = test_df[test_df["session_len"] >= 2].copy().reset_index(drop=True)

    return train_df.reset_index(drop=True), val_df, test_df


def FiltrarItemsTrainYProyectarSplits(train_df, val_df, test_df, min_session_len=2, min_item_support=5,):
    # El soporte se calcula solo con train para no meter informacion del futuro.
    train_filtrado = train_df.copy()

    while True:
        #Este filtrado se repite porque al quitar items tambien pueden caer sesiones completas,
        #y eso vuelve a cambiar el soporte de los items que quedan.
        sesiones_antes = len(train_filtrado)
        eventos_antes = int(train_filtrado["session_len"].sum())

        #Primero se quitan items demasiado raros y luego sesiones que quedaron demasiado cortas.
        soporte_items = Counter(item for secuencia in train_filtrado["items"] for item in secuencia)
        train_filtrado["items"] = train_filtrado["items"].apply(
            lambda secuencia: [item for item in secuencia if soporte_items[item] >= min_item_support])
        train_filtrado["session_len"] = train_filtrado["items"].apply(len)
        train_filtrado = train_filtrado[train_filtrado["session_len"] >= min_session_len].copy()

        sesiones_despues = len(train_filtrado)
        eventos_despues = int(train_filtrado["session_len"].sum())
        if (sesiones_antes, eventos_antes) == (sesiones_despues, eventos_despues):
            #Cuando ya no cambia nada entre iteraciones, el filtrado llego a un punto estable.
            break

    catalogo_train = set(item for secuencia in train_filtrado["items"] for item in secuencia)

    def ProyectarSplit(split_df):
        #Se proyecta cada split al catalogo final que efectivamente sobrevivio en train.
        split_proyectado = split_df.copy()
        split_proyectado["items"] = split_proyectado["items"].apply(lambda secuencia: [item for item in secuencia if item in catalogo_train])
        
        #Igual que antes, la proyeccion puede dejar sesiones demasiado cortas para evaluar.
        split_proyectado["session_len"] = split_proyectado["items"].apply(len)
        split_proyectado = split_proyectado[split_proyectado["session_len"] >= min_session_len].copy()
        return split_proyectado.reset_index(drop=True)

    val_filtrado = ProyectarSplit(val_df)
    test_filtrado = ProyectarSplit(test_df)

    return (train_filtrado.reset_index(drop=True), val_filtrado, test_filtrado,)