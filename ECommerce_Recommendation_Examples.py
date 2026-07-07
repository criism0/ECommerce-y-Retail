from __future__ import annotations
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


def _ModaValida(series, unknown_token="unknown"):
    valores_validos = series.dropna().astype("string")
    valores_validos = valores_validos[valores_validos != unknown_token]
    if valores_validos.empty:
        return unknown_token
    return valores_validos.mode().iloc[0]


def ConstruirCatalogoProductosVisual(events_df):
    #Resume metadatos legibles por producto para inspeccionar recomendaciones.
    catalogo = (
        events_df.groupby("product_id")
        .agg(
            category_code=("category_code", lambda serie: _ModaValida(serie, "unknown")),
            brand=("brand", lambda serie: _ModaValida(serie, "unknown")),
            price=("price", "median"),
            event_count=("product_id", "size"),
        )
        .reset_index()
    )

    catalogo["price"] = pd.to_numeric(catalogo["price"], errors="coerce")
    catalogo["category_root"] = (
        catalogo["category_code"].fillna("unknown").astype("string").str.split(".").str[0]
    )
    return catalogo


def FormatearEtiquetaProducto(product_id, product_lookup, include_price=False):
    meta = product_lookup.get(product_id, {})
    category_root = meta.get("category_root", "unknown")
    brand = meta.get("brand", "unknown")
    price = meta.get("price")

    partes = [f"id={product_id}", f"cat={category_root}", f"brand={brand}"]
    if include_price and pd.notna(price):
        partes.append(f"price={float(price):.2f}")
    return " | ".join(partes)


def _ContarMetadatosUtiles(product_id, product_lookup):
    meta = product_lookup.get(product_id, {})
    score = 0
    if meta.get("category_root", "unknown") != "unknown":
        score += 1
    if meta.get("brand", "unknown") != "unknown":
        score += 1
    if pd.notna(meta.get("price")):
        score += 1
    return score


def FormatearEtiquetaProductoHumana(product_id, product_lookup, max_chars=34):
    #Convierte un producto a una etiqueta breve y legible usando metadata.
    meta = product_lookup.get(product_id, {})
    category_root = str(meta.get("category_root", "unknown"))
    brand = str(meta.get("brand", "unknown"))
    price = meta.get("price")

    partes = []
    if category_root != "unknown":
        partes.append(category_root)
    if brand != "unknown":
        partes.append(brand)
    if pd.notna(price):
        partes.append(f"${float(price):.2f}")

    if not partes:
        return "producto sin metadata"

    label = " | ".join(partes)
    if len(label) > max_chars:
        label = label[: max_chars - 3] + "..."
    return label


def SeleccionarCasosEjemplo(eval_cases, recommender_fn, product_lookup, k=5):
    #Busca un caso corto, uno medio y uno largo donde el modelo base acierte el target.
    #Cuando se puede, prioriza ejemplos con metadata util para que la lectura sea interpretable.
    casos = eval_cases.reset_index(drop=True).copy()
    buckets = [
        ("corta", lambda n: n <= 3),
        ("media", lambda n: 4 <= n <= 6),
        ("larga", lambda n: n >= 7),
    ]

    seleccionados = []
    usados = set()

    for bucket_name, bucket_fn in buckets:
        candidatos_bucket = []

        for idx, row in casos.iterrows():
            prefix_len = len(row["prefix_items"])
            if idx in usados or not bucket_fn(prefix_len):
                continue

            recs = recommender_fn(row["prefix_items"], k=k, seen_items=row["seen_items"])
            if row["target_item"] not in recs:
                continue

            target_meta_score = _ContarMetadatosUtiles(row["target_item"], product_lookup)
            prefix_meta_score = sum(
                _ContarMetadatosUtiles(item, product_lookup) for item in row["prefix_items"]
            )
            candidatos_bucket.append((target_meta_score, prefix_meta_score, prefix_len, idx, row))

        if candidatos_bucket:
            _, _, prefix_len, idx, row = sorted(candidatos_bucket, reverse=True)[0]
            usados.add(idx)
            seleccionados.append(
                {
                    "case_idx": idx,
                    "bucket": bucket_name,
                    "prefix_len": prefix_len,
                    "target_item": row["target_item"],
                    "target_desc": FormatearEtiquetaProducto(
                        row["target_item"], product_lookup, include_price=True
                    ),
                }
            )

    if not seleccionados:
        #Respaldo si no se encuentra ningun hit representativo.
        for idx, row in casos.head(3).iterrows():
            seleccionados.append(
                {
                    "case_idx": idx,
                    "bucket": "fallback",
                    "prefix_len": len(row["prefix_items"]),
                    "target_item": row["target_item"],
                    "target_desc": FormatearEtiquetaProducto(
                        row["target_item"], product_lookup, include_price=True
                    ),
                }
            )

    return pd.DataFrame(seleccionados)


def ConstruirTablaSecuenciaCaso(case_row, product_lookup):
    rows = []

    for paso, item in enumerate(case_row["prefix_items"], start=1):
        rows.append(
            {
                "rol": f"prefijo_{paso}",
                "product_id": item,
                "resumen": FormatearEtiquetaProductoHumana(item, product_lookup),
                "descripcion": FormatearEtiquetaProducto(item, product_lookup, include_price=True),
            }
        )

    rows.append(
        {
            "rol": "target_real",
            "product_id": case_row["target_item"],
            "resumen": FormatearEtiquetaProductoHumana(case_row["target_item"], product_lookup),
            "descripcion": FormatearEtiquetaProducto(
                case_row["target_item"], product_lookup, include_price=True
            ),
        }
    )
    return pd.DataFrame(rows)


def ConstruirTablaRecomendacionesCaso(case_row, recommenders, product_lookup, k=5):
    rows = []
    target_item = case_row["target_item"]

    for model_name, recommender_fn in recommenders:
        recs = recommender_fn(case_row["prefix_items"], k=k, seen_items=case_row["seen_items"])
        for rank, item in enumerate(recs, start=1):
            rows.append(
                {
                    "modelo": model_name,
                    "rank": rank,
                    "product_id": item,
                    "es_target": item == target_item,
                    "resumen": FormatearEtiquetaProductoHumana(item, product_lookup),
                    "descripcion": FormatearEtiquetaProducto(item, product_lookup, include_price=True),
                }
            )

    return pd.DataFrame(rows)


def PlotComparacionRecomendacionesCaso(case_row, recommendations_df, product_lookup, k=5, figsize=(16, 7)):
    #Dibuja arriba la secuencia observada y abajo una matriz simple de recomendaciones.
    modelos = recommendations_df["modelo"].drop_duplicates().tolist()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.25)

    ax_seq = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])

    prefix_items = case_row["prefix_items"]
    target_item = case_row["target_item"]
    seq_items = prefix_items + [target_item]
    seq_roles = ["prefijo"] * len(prefix_items) + ["target"]

    ax_seq.set_xlim(-0.5, len(seq_items) - 0.5)
    ax_seq.set_ylim(-0.5, 0.5)
    ax_seq.axis("off")

    for x, (item, rol) in enumerate(zip(seq_items, seq_roles)):
        color = "#dbe9fb" if rol == "prefijo" else "#ffe6cc"
        ax_seq.add_patch(
            plt.Rectangle((x - 0.45, -0.18), 0.9, 0.36, facecolor=color, edgecolor="#123f76", linewidth=1.2)
        )
        human_label = FormatearEtiquetaProductoHumana(item, product_lookup, max_chars=28)
        ax_seq.text(
            x,
            0.04,
            human_label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#123f76",
            wrap=True,
        )
        ax_seq.text(
            x,
            -0.12,
            rol,
            ha="center",
            va="center",
            fontsize=8,
            color="#4b5f7d",
        )

    target_label = FormatearEtiquetaProductoHumana(target_item, product_lookup, max_chars=60)
    ax_seq.set_title(
        f"Secuencia observada y target real\n{target_label}",
        fontsize=13,
        fontweight="bold",
        color="#123f76",
    )

    ax_grid.set_xlim(-0.5, k - 0.5)
    ax_grid.set_ylim(-0.5, len(modelos) - 0.5)
    ax_grid.set_xticks(range(k))
    ax_grid.set_xticklabels([f"rank {idx}" for idx in range(1, k + 1)], fontsize=10)
    ax_grid.set_yticks(range(len(modelos)))
    ax_grid.set_yticklabels(modelos, fontsize=10)
    ax_grid.invert_yaxis()
    ax_grid.set_title("Top-k recomendado por modelo", fontsize=13, fontweight="bold", color="#123f76")

    for row_idx, model_name in enumerate(modelos):
        subset = recommendations_df[recommendations_df["modelo"] == model_name].sort_values("rank")
        for col_idx in range(k):
            cell = subset[subset["rank"] == col_idx + 1]
            if cell.empty:
                text = "-"
                facecolor = "#f3f5f8"
            else:
                text = str(cell.iloc[0]["resumen"])
                facecolor = "#d9f2d9" if bool(cell.iloc[0]["es_target"]) else "#edf3fb"

            ax_grid.add_patch(
                plt.Rectangle((col_idx - 0.45, row_idx - 0.35), 0.9, 0.7, facecolor=facecolor, edgecolor="#cfd7e4")
            )
            ax_grid.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color="#10233f",
                wrap=True,
            )

    ax_grid.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_grid.tick_params(length=0)
    return fig
