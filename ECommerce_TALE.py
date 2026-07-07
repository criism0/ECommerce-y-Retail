from __future__ import annotations
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


SEGUNDOS_POR_DIA = 86_400.0 #Cantidad de segundos en un día para la ventana


def _EliminarDuplicadosConsecutivosConTiempo(items, timestamps):
    #Mantiene el timestamp de la ocurrencia que efectivamente sobrevive.
    items_limpios = []
    tiempos_limpios = []

    for item, timestamp in zip(items, timestamps):
        if not items_limpios or items_limpios[-1] != item:
            items_limpios.append(item)
            tiempos_limpios.append(timestamp)

    return items_limpios, tiempos_limpios


def _ConvertirSerieTimestampsASegundos(timestamp_series):
    timestamps = pd.to_datetime(timestamp_series, utc=True, errors="coerce")
    #int64 en segundos Unix para poder operar rapido.
    return (timestamps.astype("int64") // 10**9).astype("int64")


def _FiltrarParesPorCatalogo(items, timestamps, catalogo):
    #Filtra una sesion al catalogo permitido conservando alineados items y timestamps.
    items_filtrados = []
    tiempos_filtrados = []

    for item, timestamp in zip(items, timestamps):
        if item in catalogo:
            items_filtrados.append(item)
            tiempos_filtrados.append(timestamp)

    return items_filtrados, tiempos_filtrados


def _AplicarFiltradoConTiempo(split_df, catalogo, min_session_len):
    #Proyecta todo un split al catalogo valido y descarta sesiones que quedan demasiado cortas.
    split_filtrado = split_df.copy()
    if split_filtrado.empty:
        split_filtrado["session_len"] = pd.Series(dtype="int64")
        return split_filtrado.reset_index(drop=True)

    pares_filtrados = [
        _FiltrarParesPorCatalogo(fila.items, fila.timestamps, catalogo)
        for fila in split_filtrado.itertuples(index=False)
    ]
    split_filtrado["items"] = [par[0] for par in pares_filtrados]
    split_filtrado["timestamps"] = [par[1] for par in pares_filtrados]
    split_filtrado["session_len"] = split_filtrado["items"].apply(len)
    split_filtrado = split_filtrado[split_filtrado["session_len"] >= min_session_len].copy()
    return split_filtrado.reset_index(drop=True)


def ConstruirSesionesConTiempo(events_df):
    #Agrupa eventos por sesion y conserva secuencias de items + timestamps. Requiere al menos user_session product_id y event_time
    columnas_requeridas = {"user_session", "product_id", "event_time"}
    faltantes = columnas_requeridas.difference(events_df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para TALE: {sorted(faltantes)}")

    eventos = events_df.copy()
    eventos["event_time"] = pd.to_datetime(eventos["event_time"], utc=True, errors="coerce")
    eventos = eventos.dropna(subset=["user_session", "product_id", "event_time"]).copy()
    eventos["product_id"] = pd.to_numeric(eventos["product_id"], errors="coerce")
    eventos = eventos.dropna(subset=["product_id"]).copy()
    eventos["product_id"] = eventos["product_id"].astype("int64")
    eventos = eventos.sort_values(["user_session", "event_time"]).reset_index(drop=True)
    eventos["timestamp_sec"] = _ConvertirSerieTimestampsASegundos(eventos["event_time"])

    agregaciones = {
        "items": ("product_id", list),
        "timestamps": ("timestamp_sec", list),
        "end_time": ("event_time", "max"),
    }
    if "user_id" in eventos.columns:
        agregaciones["user_id"] = ("user_id", "last")

    sesiones = eventos.groupby("user_session").agg(**agregaciones).reset_index()

    secuencias_limpias = sesiones.apply(
        lambda fila: _EliminarDuplicadosConsecutivosConTiempo(fila["items"], fila["timestamps"]),
        axis=1,
    )
    sesiones["items"] = secuencias_limpias.apply(lambda par: par[0])
    sesiones["timestamps"] = secuencias_limpias.apply(lambda par: par[1])
    sesiones["session_len"] = sesiones["items"].apply(len)
    return sesiones


def SepararSesionesPorVentanasDeTiempoTALE(session_df, validation_days=3, test_days=3):
    #Replica el criterio temporal del pipeline principal pero conservando timestamps.
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days y test_days deben ser enteros positivos.")

    sesiones_ordenadas = session_df.sort_values("end_time").reset_index(drop=True)
    fechas_fin = sesiones_ordenadas["end_time"].dt.normalize()
    fechas_unicas = fechas_fin.drop_duplicates().sort_values().reset_index(drop=True)

    dias_necesarios = validation_days + test_days + 1
    if len(fechas_unicas) < dias_necesarios:
        raise ValueError("No hay suficientes fechas distintas para crear ventanas de train/validation/test.")

    inicio_test = fechas_unicas.iloc[-test_days]
    inicio_validacion = fechas_unicas.iloc[-(test_days + validation_days)]

    train_df = sesiones_ordenadas[fechas_fin < inicio_validacion].copy()
    val_df = sesiones_ordenadas[(fechas_fin >= inicio_validacion) & (fechas_fin < inicio_test)].copy()
    test_df = sesiones_ordenadas[fechas_fin >= inicio_test].copy()

    catalogo_train = set(item for secuencia in train_df["items"] for item in secuencia)
    val_df = _AplicarFiltradoConTiempo(val_df, catalogo_train, min_session_len=2)
    test_df = _AplicarFiltradoConTiempo(test_df, catalogo_train, min_session_len=2)

    return train_df.reset_index(drop=True), val_df, test_df


def FiltrarItemsTrainYProyectarSplitsTALE(train_df, val_df, test_df, min_session_len=2, min_item_support=5):
    #Filtra items usando solo train y proyecta validation/test al catalogo final.
    train_filtrado = train_df.copy()

    while True:
        if train_filtrado.empty:
            break

        sesiones_antes = len(train_filtrado)
        eventos_antes = int(train_filtrado["session_len"].sum())

        soporte_items = Counter(item for secuencia in train_filtrado["items"] for item in secuencia)

        pares_filtrados = [
            [
                (item, timestamp)
                for item, timestamp in zip(fila.items, fila.timestamps)
                if soporte_items[item] >= min_item_support
            ]
            for fila in train_filtrado.itertuples(index=False)
        ]
        train_filtrado["items"] = [[item for item, _ in pares] for pares in pares_filtrados]
        train_filtrado["timestamps"] = [[timestamp for _, timestamp in pares] for pares in pares_filtrados]
        train_filtrado["session_len"] = train_filtrado["items"].apply(len)
        train_filtrado = train_filtrado[train_filtrado["session_len"] >= min_session_len].copy()

        sesiones_despues = len(train_filtrado)
        eventos_despues = int(train_filtrado["session_len"].sum())
        if (sesiones_antes, eventos_antes) == (sesiones_despues, eventos_despues):
            break

    catalogo_train = set(item for secuencia in train_filtrado["items"] for item in secuencia)
    val_filtrado = _AplicarFiltradoConTiempo(val_df, catalogo_train, min_session_len=min_session_len)
    test_filtrado = _AplicarFiltradoConTiempo(test_df, catalogo_train, min_session_len=min_session_len)

    return train_filtrado.reset_index(drop=True), val_filtrado, test_filtrado


def LimitarCatalogoTALEPorPopularidad(train_df, val_df, test_df, max_catalog_items=None, min_session_len=2):
    #Limita train/validation/test al top-N de train para que la solucion cerrada
    #de TALE sea manejable en memoria y tiempo.
    if max_catalog_items is None:
        return train_df.copy(), val_df.copy(), test_df.copy(), None

    soporte_items = Counter(item for secuencia in train_df["items"] for item in secuencia)
    top_items = {
        item for item, _ in soporte_items.most_common(max_catalog_items)
    }

    train_limitado = _AplicarFiltradoConTiempo(train_df, top_items, min_session_len=min_session_len)
    val_limitado = _AplicarFiltradoConTiempo(val_df, top_items, min_session_len=min_session_len)
    test_limitado = _AplicarFiltradoConTiempo(test_df, top_items, min_session_len=min_session_len)

    return train_limitado, val_limitado, test_limitado, top_items


def ConstruirResumenSplitsTALE(train_df, val_df, test_df):
    #Resume tamaño, cobertura temporal y diversidad de items en cada split de TALE.
    def Resumen(nombre, df):
        return {
            "split": nombre,
            "sesiones": len(df),
            "interacciones": int(df["session_len"].sum()) if not df.empty else 0,
            "items_unicos": len({item for secuencia in df["items"] for item in secuencia}),
            "inicio": df["end_time"].min() if not df.empty else pd.NaT,
            "fin": df["end_time"].max() if not df.empty else pd.NaT,
        }

    return pd.DataFrame(
        [Resumen("train", train_df), Resumen("validation", val_df), Resumen("test", test_df)]
    )


def _NormalizarInputSesiones(sessions):
    #Lleva distintas entradas de sesiones a un formato comun con items, timestamps y longitud valida.
    if isinstance(sessions, pd.DataFrame):
        normalizado = sessions.copy()
    else:
        normalizado = pd.DataFrame(list(sessions))

    if "items" not in normalizado.columns:
        raise ValueError("Las sesiones para TALE deben incluir la columna 'items'.")

    if "timestamps" not in normalizado.columns:
        #Respaldo para compatibilidad: usa el orden relativo si no hay timestamps.
        normalizado["timestamps"] = normalizado["items"].apply(lambda secuencia: list(range(len(secuencia))))

    normalizado["items"] = normalizado["items"].apply(list)
    normalizado["timestamps"] = normalizado["timestamps"].apply(list)
    normalizado["session_len"] = normalizado["items"].apply(len)
    normalizado = normalizado[normalizado["session_len"] >= 2].copy().reset_index(drop=True)
    return normalizado


class TALERecommender:
    #Recomendador TALE.
    
    #Inicializa hiperparametros de entrenamiento, decaimiento temporal
    #y restricciones prácticas de catálogo para mantener viable la forma cerrada.
    def __init__(
        self,
        lambda_reg=200.0,
        c=0.2,
        tau_time=21_600.0,
        window_days=7,
        gamma=0.5,
        tau_inference=2.0,
        max_catalog_items=2_000,
        trend_use_future=True,
        verbose=False,
        reg_weight=None,
        tau_train=None,
    ):
        if reg_weight is not None:
            lambda_reg = reg_weight
        if tau_train is not None:
            tau_time = tau_train

        if lambda_reg <= 0:
            raise ValueError("lambda_reg debe ser positivo.")
        if not 0 <= c <= 1:
            raise ValueError("c debe estar entre 0 y 1.")
        if tau_time <= 0:
            raise ValueError("tau_time debe ser positivo.")
        if tau_inference <= 0:
            raise ValueError("tau_inference debe ser positivo.")
        if window_days <= 0:
            raise ValueError("window_days debe ser positivo.")

        self.lambda_reg = float(lambda_reg)
        self.c = float(c)
        self.tau_time = float(tau_time)
        self.window_days = int(window_days)
        self.gamma = float(gamma)
        self.tau_inference = float(tau_inference)
        self.max_catalog_items = max_catalog_items
        self.trend_use_future = bool(trend_use_future)
        self.verbose = bool(verbose)

        self.B_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.popular_items = []
        self.item_support = Counter()
        self.item_timestamp_index = {}
        self.catalogo_entrenamiento = set()

    def _ConstruirMapeoItems(self, train_sessions):
        #Define el catalogo final de TALE y crea indices compactos item -> columna.
        soporte = Counter(item for secuencia in train_sessions["items"] for item in secuencia)
        if self.max_catalog_items is None:
            items_ordenados = [item for item, _ in soporte.most_common()]
        else:
            items_ordenados = [item for item, _ in soporte.most_common(self.max_catalog_items)]

        self.catalogo_entrenamiento = set(items_ordenados)
        self.item_support = Counter({item: soporte[item] for item in items_ordenados})
        self.popular_items = items_ordenados[:]
        self.item_to_idx = {item: idx for idx, item in enumerate(items_ordenados)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

    def _PrepararIndiceTemporalPorItem(self, train_sessions):
        #Guarda los timestamps por item para estimar su popularidad local en ventanas de tiempo.
        timestamps_por_item = defaultdict(list)

        for fila in train_sessions.itertuples(index=False):
            for item, timestamp in zip(fila.items, fila.timestamps):
                if item in self.catalogo_entrenamiento:
                    timestamps_por_item[item].append(int(timestamp))

        self.item_timestamp_index = {
            item: np.array(sorted(timestamps), dtype=np.int64)
            for item, timestamps in timestamps_por_item.items()
        }

    def _ContarPopularidadLocal(self, item, timestamp_referencia):
        #Cuenta cuantas veces aparece un item cerca del instante de referencia.
        timestamps = self.item_timestamp_index.get(item)
        if timestamps is None or len(timestamps) == 0:
            return 1

        ventana_segundos = int(self.window_days * SEGUNDOS_POR_DIA)
        if self.trend_use_future:
            limite_inferior = timestamp_referencia - ventana_segundos
            limite_superior = timestamp_referencia + ventana_segundos
        else:
            limite_inferior = timestamp_referencia - ventana_segundos
            limite_superior = timestamp_referencia

        left = bisect_left(timestamps, limite_inferior)
        right = bisect_right(timestamps, limite_superior)
        return max(right - left, 1)

    def _PesoIntervaloTemporal(self, source_time, target_time):
        #Penaliza dependencias lejanas en el tiempo con un decaimiento exponencial acotado por c.
        delta = max(float(target_time) - float(source_time), 0.0)
        return max(math.exp(-delta / self.tau_time), self.c)

    def _PesoTendencia(self, item, target_time):
        #Ajusta el peso de un item segun su popularidad reciente alrededor del target.
        popularidad_local = self._ContarPopularidadLocal(item, target_time)
        return popularidad_local ** (-self.gamma)

    def _ConstruirMatricesEntrenamiento(self, train_sessions):
        #Construye las matrices sparse S y T de TALE a partir de prefijos y siguientes items.
        filas_s = []
        columnas_s = []
        datos_s = []
        filas_t = []
        columnas_t = []
        datos_t = []

        fila_actual = 0

        for fila in train_sessions.itertuples(index=False):
            pares_filtrados = [
                (item, int(timestamp))
                for item, timestamp in zip(fila.items, fila.timestamps)
                if item in self.catalogo_entrenamiento
            ]

            if len(pares_filtrados) < 2:
                continue

            items = [item for item, _ in pares_filtrados]
            tiempos = [timestamp for _, timestamp in pares_filtrados]

            for fin in range(1, len(items)):
                target_item = items[fin]
                target_time = tiempos[fin]
                target_idx = self.item_to_idx[target_item]

                peso_target = self._PesoTendencia(target_item, target_time)
                filas_t.append(fila_actual)
                columnas_t.append(target_idx)
                datos_t.append(peso_target)

                pesos_por_item = {}
                for source_item, source_time in zip(items[:fin], tiempos[:fin]):
                    source_idx = self.item_to_idx[source_item]
                    peso_temporal = self._PesoIntervaloTemporal(source_time, target_time)
                    peso_tendencia = self._PesoTendencia(source_item, target_time)
                    peso_total = peso_temporal * peso_tendencia

                    peso_prev = pesos_por_item.get(source_idx, 0.0)
                    if peso_total > peso_prev:
                        pesos_por_item[source_idx] = peso_total

                for source_idx, peso_total in pesos_por_item.items():
                    filas_s.append(fila_actual)
                    columnas_s.append(source_idx)
                    datos_s.append(peso_total)

                fila_actual += 1

        if fila_actual == 0:
            raise ValueError("TALE no pudo generar ejemplos de entrenamiento con las sesiones entregadas.")

        num_items = len(self.item_to_idx)
        S = csr_matrix((datos_s, (filas_s, columnas_s)), shape=(fila_actual, num_items), dtype=np.float64)
        T = csr_matrix((datos_t, (filas_t, columnas_t)), shape=(fila_actual, num_items), dtype=np.float64)
        return S, T

    def fit(self, sessions_df):
        #Entrena la matriz item-item resolviendo la forma cerrada regularizada de TALE.
        train_sessions = _NormalizarInputSesiones(sessions_df)
        self._ConstruirMapeoItems(train_sessions)
        train_sessions = _AplicarFiltradoConTiempo(
            train_sessions,
            self.catalogo_entrenamiento,
            min_session_len=2,
        )
        self._PrepararIndiceTemporalPorItem(train_sessions)
        S, T = self._ConstruirMatricesEntrenamiento(train_sessions)

        if self.verbose:
            print(
                f"TALE: sesiones={len(train_sessions):,}, ejemplos={S.shape[0]:,}, "
                f"items={S.shape[1]:,}, lambda={self.lambda_reg:.3f}"
            )

        gram = (S.T @ S).toarray()
        rhs = (S.T @ T).toarray()
        gram += self.lambda_reg * np.eye(gram.shape[0], dtype=np.float64)

        self.B_matrix = np.linalg.solve(gram, rhs).astype(np.float32)
        return self

    def _ConstruirVectorInferencia(self, prefix_items):
        #Convierte el prefijo observado en un vector de inferencia donde pesa mas lo reciente.
        if self.B_matrix is None:
            raise ValueError("El modelo TALE aun no fue entrenado.")

        vector = np.zeros(len(self.item_to_idx), dtype=np.float32)
        if not prefix_items:
            return vector

        items_validos = [item for item in prefix_items if item in self.item_to_idx]
        if not items_validos:
            return vector

        for distancia, item in enumerate(reversed(items_validos), start=1):
            idx = self.item_to_idx[item]
            peso = math.exp(-(distancia - 1) / self.tau_inference)
            if peso > vector[idx]:
                vector[idx] = peso

        return vector

    def Puntuar(self, prefix_items, seen_items=None):
        #Calcula scores para los items candidatos excluyendo los ya vistos.
        vistos = set(prefix_items) if seen_items is None else set(seen_items)
        vector = self._ConstruirVectorInferencia(prefix_items)
        if not vector.any():
            return {}

        scores = vector @ self.B_matrix
        puntajes = {}
        for idx, score in enumerate(scores):
            item = self.idx_to_item[idx]
            if item in vistos:
                continue
            puntajes[item] = float(score)
        return puntajes

    def Recomendar(self, prefix_items, k=10, seen_items=None):
        #Ordena los candidatos por score y completa con populares si faltan recomendaciones.
        vistos = set(prefix_items) if seen_items is None else set(seen_items)
        scores = self.Puntuar(prefix_items, seen_items=vistos)
        recomendaciones = [
            item for item, _ in sorted(scores.items(), key=lambda par: par[1], reverse=True)
        ]

        if len(recomendaciones) < k:
            for item in self.popular_items:
                if item in vistos or item in recomendaciones:
                    continue
                recomendaciones.append(item)
                if len(recomendaciones) == k:
                    break

        return recomendaciones[:k]

    def score(self, prefix_items):
        #Compatibilidad con implementaciones previas.
        return self.Puntuar(prefix_items)

    def recommend(self, prefix_items, k=10, exclude_items=None):
        #Alias de compatibilidad para usar la interfaz estilo recommend.
        return self.Recomendar(prefix_items, k=k, seen_items=exclude_items)


class TALEExperimenter:
    #Wrapper liviano mantenido por compatibilidad con scripts previos.
    def __init__(self, **kwargs):
        self.model = TALERecommender(**kwargs)
        self.name = "TALE"

    def fit(self, sessions_df):
        self.model.fit(sessions_df)
        return self

    def score(self, prefix_items):
        return self.model.score(prefix_items)

    def recommend(self, prefix_items, k=10, exclude_items=None):
        return self.model.recommend(prefix_items, k=k, exclude_items=exclude_items)


def EntrenarTALE(train_sessions, **kwargs):
    recomendador = TALERecommender(**kwargs)
    recomendador.fit(train_sessions)
    return recomendador


def HiperparametrosTALE():
    #Configuraciones iniciales pensadas para este dataset de e-commerce.
    una_hora = 3_600.0
    return {
        "balanced": {
            "lambda_reg": 200.0,
            "c": 0.20,
            "tau_time": 6 * una_hora,
            "window_days": 7,
            "tau_inference": 2.0,
            "max_catalog_items": 2_000,
        },
        "short_term": {
            "lambda_reg": 100.0,
            "c": 0.10,
            "tau_time": 2 * una_hora,
            "window_days": 3,
            "tau_inference": 1.5,
            "max_catalog_items": 2_000,
        },
        "long_term": {
            "lambda_reg": 300.0,
            "c": 0.30,
            "tau_time": 24 * una_hora,
            "window_days": 14,
            "tau_inference": 3.0,
            "max_catalog_items": 2_000,
        },
    }


def EvaluarTALE(train_sessions, test_sessions, hparams=None, ks=(5, 10, 20)):
    #Evaluacion simple fuera del notebook principal.
    if hparams is None:
        hparams = HiperparametrosTALE()["balanced"]

    tale = EntrenarTALE(train_sessions, **hparams)
    test_df = _NormalizarInputSesiones(test_sessions)
    rows = []

    for k in ks:
        recall_scores = []
        ndcg_scores = []
        map_scores = []

        for fila in test_df.itertuples(index=False):
            prefix = fila.items[:-1]
            target = fila.items[-1]
            recs = tale.Recomendar(prefix, k=max(ks), seen_items=set(prefix))

            hit = 1.0 if target in recs[:k] else 0.0
            recall_scores.append(hit)

            if target in recs[:k]:
                rank = recs.index(target) + 1
                ndcg_scores.append(1.0 / np.log2(rank + 1))
                map_scores.append(1.0 / rank)
            else:
                ndcg_scores.append(0.0)
                map_scores.append(0.0)

        rows.append(
            {
                "modelo": "tale",
                "k": k,
                "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
                "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
                "map": float(np.mean(map_scores)) if map_scores else 0.0,
            }
        )

    return pd.DataFrame(rows)
