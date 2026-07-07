import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def _NormalizarScores(scores):
    #Antes de mezclar modelos, hay que llevar sus scores a una escala comparable.
    if not scores:
        return {}

    puntajes = list(scores.values())
    score_max = max(puntajes)
    score_min = min(puntajes)

    if math.isclose(score_max, score_min):
        #Si todos los items tienen el mismo score, se dejan empatados.
        return {item: 1.0 for item in scores}

    rango = score_max - score_min
    return {item: (score - score_min) / rango for item, score in scores.items()}

#Clase que aporta una forma comun de puntuar y recomendar.
class RecomendadorPorScores:
    def __init__(self, popular_items, generador_scores):
        self.popular_items = popular_items
        self.generador_scores = generador_scores

    def Puntuar(self, prefix_items, seen_items=None):
        #Si no se pasa otra lista, se asume que lo visto es lo que ya viene en el prefijo.
        vistos = set(prefix_items) if seen_items is None else set(seen_items)
        return self.generador_scores(prefix_items, vistos)

    def Recomendar(self, prefix_items, k=10, seen_items=None):
        vistos = set(prefix_items) if seen_items is None else set(seen_items)
        scores = self.Puntuar(prefix_items, seen_items=vistos)
        #Primero se ordenan solo los candidatos que el modelo realmente pudo puntuar.
        recomendaciones = [item for item, _ in sorted(scores.items(), key=lambda par: par[1], reverse=True)]

        #Si el modelo devuelve pocos candidatos, se rellena con populares no vistos.
        if len(recomendaciones) < k:
            for item in self.popular_items:
                if item in vistos or item in recomendaciones:
                    continue
                recomendaciones.append(item)
                if len(recomendaciones) == k:
                    break

        return recomendaciones[:k]


def ConstruirItemKnnRecomendador(train_sessions, max_neighbors=100, window_size=3, last_n=5,):
    #La idea es simple, si dos items suelen verse cerca en una sesion, probablemente esten relacionados.
    soporte_items = Counter()
    coocurrencias = defaultdict(Counter)

    for secuencia in train_sessions["items"]:
        #Se quitan duplicados conservando el orden para no inflar la sesion artificialmente.
        secuencia_unica = list(dict.fromkeys(secuencia))

        for item in secuencia_unica:
            #El soporte cuenta en cuantas sesiones aparece el item, no cuantas veces se repite.
            soporte_items[item] += 1

        for posicion, item in enumerate(secuencia_unica):
            #Solo se miran vecinos cercanos porque este modelo busca contexto local dentro de la sesion.
            inicio_ventana = max(0, posicion - window_size)
            fin_ventana = min(len(secuencia_unica), posicion + window_size + 1)
            for otra_posicion in range(inicio_ventana, fin_ventana):
                if posicion == otra_posicion:
                    continue
                vecino = secuencia_unica[otra_posicion]
                coocurrencias[item][vecino] += 1

    vecinos_por_item = {}
    for item, vecinos in coocurrencias.items():
        vecinos_con_score = []
        for vecino, conteo in vecinos.items():
            #La coocurrencia se normaliza para no favorecer tanto a items que son populares con todo.
            similitud = conteo / math.sqrt(soporte_items[item] * soporte_items[vecino])
            vecinos_con_score.append((vecino, similitud))

        #Nos quedamos solo con los vecinos mas utiles de cada item.
        vecinos_por_item[item] = sorted(vecinos_con_score, key=lambda par: par[1], reverse=True,)[:max_neighbors]

    items_populares = [item for item, _ in soporte_items.most_common()]

    def GenerarScores(prefix_items, seen_items):
        historial_reciente = prefix_items[-last_n:]
        scores = Counter()

        #Los items mas recientes de la sesion pesan mas que los antiguos.
        for orden, item in enumerate(reversed(historial_reciente), start=1):
            peso = 1 / orden
            for vecino, similitud in vecinos_por_item.get(item, []):
                if vecino in seen_items:
                    continue
                scores[vecino] += similitud * peso

        return dict(scores)

    return RecomendadorPorScores(items_populares, GenerarScores)


def ConstruirSequentialRulesRecomendador(train_sessions, max_steps=10, max_rules_per_item=200, last_n=5,):
    #Aqui se aprende algo mas parecido a reglas de transicion, si aparece A, luego suele venir B.
    reglas = defaultdict(Counter)
    popularidad = Counter()

    for secuencia in train_sessions["items"]:
        secuencia_unica = list(dict.fromkeys(secuencia))

        for item in secuencia_unica:
            popularidad[item] += 1

        for posicion, item in enumerate(secuencia_unica[:-1]):
            #Desde cada item se mira algunos pasos hacia adelante para aprender transiciones cortas.
            limite = min(len(secuencia_unica), posicion + 1 + max_steps)
            for siguiente_posicion in range(posicion + 1, limite):
                siguiente_item = secuencia_unica[siguiente_posicion]
                #Mientras mas cerca esta la transicion, mas peso recibe.
                distancia = siguiente_posicion - posicion
                reglas[item][siguiente_item] += 1 / distancia

    top_reglas = {}
    for item, vecinos in reglas.items():
        top_reglas[item] = sorted(vecinos.items(), key=lambda par: par[1], reverse=True,)[:max_rules_per_item]

    items_populares = [item for item, _ in popularidad.most_common()]

    def GenerarScores(prefix_items, seen_items):
        historial_reciente = prefix_items[-last_n:]
        scores = Counter()

        #Se reutiliza la misma idea de recencia, lo ultimo visto suele importar mas.
        for orden, item in enumerate(reversed(historial_reciente), start=1):
            peso = 1 / orden
            for vecino, score_regla in top_reglas.get(item, []):
                if vecino in seen_items:
                    continue
                scores[vecino] += score_regla * peso

        return dict(scores)

    return RecomendadorPorScores(items_populares, GenerarScores)


def ConstruirRecomendadorHibrido(recommenders, popular_items):
    #El hibrido mezcla modelos distintos sin asumir que todos puntuan en la misma escala.
    def GenerarScores(prefix_items, seen_items):
        scores_combinados = Counter()

        for _, recommender, peso in recommenders:
            if hasattr(recommender, "Puntuar"):
                #Si el recomendador ya da scores, se usan tal cual antes de normalizar.
                scores_modelo = recommender.Puntuar(prefix_items, seen_items=seen_items)
            else:
                #Si solo entrega un ranking, se crea un score simple basado en la posicion.
                recomendaciones = recommender.Recomendar(prefix_items, k=100, seen_items=seen_items,)
                scores_modelo = {item: 1 / (posicion + 1) for posicion, item in enumerate(recomendaciones)}

            #La suma ponderada solo tiene sentido despues de normalizar cada modelo por separado.
            for item, score in _NormalizarScores(scores_modelo).items():
                scores_combinados[item] += peso * score

        return dict(scores_combinados)

    return RecomendadorPorScores(popular_items, GenerarScores)


def ConstruirRecomendadorPopularidadReciente(events_df, recent_days=7, event_weights=None, half_life_days=3.0,):
    #No todos los eventos valen igual, y tampoco da lo mismo que hayan pasado hace dos horas o hace una semana.
    if event_weights is None:
        event_weights = {"view": 1.0, "cart": 3.0, "purchase": 5.0}

    fecha_max = events_df["event_time"].max()
    fecha_corte = fecha_max - pd.Timedelta(days=recent_days)
    eventos_recientes = events_df[events_df["event_time"] >= fecha_corte].copy()

    if eventos_recientes.empty:
        #Si no hay eventos en esa ventana, se deja un respaldo simple para no romper el flujo.
        items_populares = events_df["product_id"].value_counts().index.tolist()
        return RecomendadorPorScores(items_populares, lambda prefix_items, seen_items: {})

    edad_en_dias = (fecha_max - eventos_recientes["event_time"]).dt.total_seconds() / 86400.0
    eventos_recientes["peso_evento"] = eventos_recientes["event_type"].map(event_weights).fillna(1.0)
    #La recencia se modela con un decaimiento suave.
    eventos_recientes["peso_recencia"] = np.exp(-edad_en_dias.clip(lower=0) / max(half_life_days, 1e-6))
    eventos_recientes["score"] = eventos_recientes["peso_evento"] * eventos_recientes["peso_recencia"]

    scores_popularidad = (eventos_recientes.groupby("product_id")["score"].sum().sort_values(ascending=False))
    scores_por_item = scores_popularidad.to_dict()
    items_populares = scores_popularidad.index.tolist()

    def GenerarScores(prefix_items, seen_items):
        return {item: score for item, score in scores_por_item.items() if item not in seen_items}

    return RecomendadorPorScores(items_populares, GenerarScores)


def ConstruirMetadatosItems(events_df):
    #Para cada producto se resume una categoria, una marca y un precio representativos.
    def ModaValida(series, unknown_token="unknown"):
        #Se ignoran nulos y "unknown" para quedarse, cuando se puede, con una señal mas util.
        valores_validos = series.dropna()
        valores_validos = valores_validos[valores_validos != unknown_token]
        if valores_validos.empty:
            return unknown_token
        return valores_validos.mode().iloc[0]

    metadatos_items = (events_df.groupby("product_id").agg(category_code=("category_code", lambda serie: ModaValida(serie, "unknown")), brand=("brand", lambda serie: ModaValida(serie, "unknown")), price=("price", "median"),).reset_index())

    #El log del precio deja comparaciones de precio un poco mas estables que el valor integro.
    metadatos_items["price"] = metadatos_items["price"].fillna(metadatos_items["price"].median())
    metadatos_items["log_price"] = np.log1p(metadatos_items["price"].clip(lower=0))
    
    #La categoria raiz sirve para capturar cercania semantica aunque no coincida toda la ruta.
    metadatos_items["category_root"] = (metadatos_items["category_code"].fillna("unknown").astype(str).str.split(".").str[0])
    return metadatos_items


def ConstruirRecomendadorRerankMetadatos(base_recommender, item_metadata, popular_items, candidate_k=100, weight_base=0.70, weight_category=0.15, weight_brand=0.05, weight_price=0.10,):
    #El reranking no inventa candidatos nuevos, solo reordena los que ya venian bien posicionados.
    metadatos_por_item = item_metadata.set_index("product_id").to_dict(orient="index")

    def SimilaridadPrecio(item_a, item_b):
        meta_a = metadatos_por_item.get(item_a)
        meta_b = metadatos_por_item.get(item_b)
        if meta_a is None or meta_b is None:
            return 0.0
        #Mientras mas cerca estan en log-price, mas parecido en terminos comerciales se consideran.
        return math.exp(-abs(meta_a["log_price"] - meta_b["log_price"]))

    def GenerarScores(prefix_items, seen_items):
        candidatos = base_recommender.Recomendar(prefix_items, k=candidate_k, seen_items=seen_items,)
        if not candidatos:
            return {}

        #Se toma el ultimo item reconocible de la sesion como referencia para comparar metadatos.
        ultimo_item = next((item for item in reversed(prefix_items) if item in metadatos_por_item), None,)
        ultimo_meta = metadatos_por_item.get(ultimo_item)
        total_candidatos = len(candidatos)
        scores = {}

        for posicion, item in enumerate(candidatos, start=1):
            meta_item = metadatos_por_item.get(item)
            #El ranking original sigue importando, el reranking solo lo ajusta.
            score_base = 1 - ((posicion - 1) / max(total_candidatos - 1, 1))

            score_categoria = 0.0
            score_marca = 0.0
            score_precio = 0.0

            if ultimo_meta is not None and meta_item is not None:
                #Coincidir en categoria exacta vale mas que coincidir solo en la categoria raiz.
                if (meta_item["category_code"] == ultimo_meta["category_code"] and meta_item["category_code"] != "unknown"):
                    score_categoria = 1.0
                elif (meta_item["category_root"] == ultimo_meta["category_root"] and meta_item["category_root"] != "unknown"):
                    score_categoria = 0.5

                if meta_item["brand"] == ultimo_meta["brand"] and meta_item["brand"] != "unknown":
                    score_marca = 1.0

                score_precio = SimilaridadPrecio(item, ultimo_item)

            #El score final mezcla ranking base con coherencia semantica y comercial.
            scores[item] = (weight_base * score_base + weight_category * score_categoria + weight_brand * score_marca + weight_price * score_precio)

        return scores

    return RecomendadorPorScores(popular_items, GenerarScores)