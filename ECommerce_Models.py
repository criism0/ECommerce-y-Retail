import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

#Funcion que permit que las corridas sean repetibles.
def FijarSemilla(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ConstruirMapeoItems(train_sessions):
    #Los modelos trabajan mejor con indices compactos que con ids reales de producto.
    items = sorted({item for secuencia in train_sessions["items"] for item in secuencia})
    item_to_idx = {item: idx for idx, item in enumerate(items, start=1)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    return item_to_idx, idx_to_item


def ConstruirEjemplosSecuenciales(session_df, item_to_idx, max_session_len=20, usar_todos_los_prefijos=True, max_train_examples=None, seed=42,):
    #De cada sesion se extraen pares (prefijo, siguiente item) para entrenar next-item prediction.
    ejemplos = []

    for secuencia in session_df["items"]:
        #Solo se conservan items que existan en el mapeo final de train.
        secuencia_idx = [item_to_idx[item] for item in secuencia if item in item_to_idx]
        if len(secuencia_idx) < 2:
            continue

        posiciones_finales = (range(1, len(secuencia_idx)) if usar_todos_los_prefijos else [len(secuencia_idx) - 1])

        for fin in posiciones_finales:
            #El prefijo se recorta para no dejar secuencias demasiado largas al modelo.
            prefijo = secuencia_idx[max(0, fin - max_session_len):fin]
            target = secuencia_idx[fin]
            if prefijo:
                ejemplos.append((prefijo, target))

    if max_train_examples is not None and len(ejemplos) > max_train_examples:
        #Cuando hay demasiados ejemplos, se submuestrea de forma reproducible.
        rng = random.Random(seed)
        ejemplos = rng.sample(ejemplos, max_train_examples)

    return ejemplos


class DatasetSecuencial(Dataset):
    def __init__(self, ejemplos):
        self.ejemplos = ejemplos

    def __len__(self):
        return len(self.ejemplos)

    def __getitem__(self, idx):
        return self.ejemplos[idx]


def CollateSecuencial(batch):
    #Como las sesiones tienen largo variable, se rellenan para armar el batch.
    prefijos, targets = zip(*batch)
    largos = torch.tensor([len(prefijo) for prefijo in prefijos], dtype=torch.long)
    largo_max = int(largos.max().item())

    #El padding se hace con ceros porque el embedding usa 0 como indice reservado.
    batch_padded = torch.zeros((len(prefijos), largo_max), dtype=torch.long)
    for fila, prefijo in enumerate(prefijos):
        batch_padded[fila, : len(prefijo)] = torch.tensor(prefijo, dtype=torch.long)

    return batch_padded, largos, torch.tensor(targets, dtype=torch.long)


def TomarUltimoEstadoValido(outputs, lengths):
    #De cada secuencia se toma el ultimo estado real, no el padding del final.
    indices_batch = torch.arange(outputs.size(0), device=outputs.device)
    ultima_posicion = lengths.to(outputs.device) - 1
    return outputs[indices_batch, ultima_posicion]


class ModeloGru4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, input_ids, lengths):
        #GRU4Rec resume el prefijo en un estado oculto y desde ahi predice el siguiente item.
        embeddings = self.dropout(self.embedding(input_ids))
        salidas, _ = self.gru(embeddings)
        ultimo_estado = self.dropout(TomarUltimoEstadoValido(salidas, lengths))
        return self.output(ultimo_estado)


class ModeloSasRec(nn.Module): 
    def __init__(self, num_items, max_len=20, embedding_dim=64, num_heads=2, num_layers=2, dropout=0.2,):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        capa_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True, activation="gelu",)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, num_items + 1)

    def forward(self, input_ids, lengths):
        #SASRec mezcla embedding del item y de posicion, y usa mascara para no mirar el futuro.
        largo_secuencia = input_ids.size(1)
        posiciones = torch.arange(largo_secuencia, device=input_ids.device).unsqueeze(0)
        embeddings = self.item_embedding(input_ids) + self.position_embedding(posiciones)
        embeddings = self.dropout(embeddings)

        #La mascara evita que una posicion vea items que vienen despues en la secuencia.
        mascara_causal = torch.triu(torch.ones(largo_secuencia, largo_secuencia, device=input_ids.device, dtype=torch.bool), diagonal=1,)
        #La mascara de padding evita que el Transformer use los ceros agregados al batch.
        mascara_padding = input_ids.eq(0)
        codificado = self.encoder(embeddings, mask=mascara_causal, src_key_padding_mask=mascara_padding,)
        codificado = self.norm(codificado)
        ultimo_estado = TomarUltimoEstadoValido(codificado, lengths)
        return self.output(ultimo_estado)


def EntrenarModeloSecuencial(model, train_examples, epochs=2, batch_size=256, learning_rate=1e-3, weight_decay=1e-5, device="cpu", verbose=True,):
    #Bloque comun a GRU4Rec y SASRec.
    dataset = DatasetSecuencial(train_examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateSecuencial,)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perdida_total = 0.0
        total_ejemplos = 0

        for input_ids, lengths, targets in loader:
            #Todo el batch se mueve al dispositivo antes del forward.
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            #Ciclo clasico de entrenamiento, forward, loss, backward y update.
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            ejemplos_batch = targets.size(0)
            perdida_total += float(loss.item()) * ejemplos_batch
            total_ejemplos += ejemplos_batch

        perdida_promedio = perdida_total / max(total_ejemplos, 1)
        history.append({"epoch": epoch, "loss": perdida_promedio})

        if verbose:
            print(f"Epoch {epoch}/{epochs} - loss: {perdida_promedio:.4f}")

    return history


class RecomendadorSecuencialNeuronal:
    #El modelo recibe el prefijo de la sesion y devuelve los top-k no vistos.
    def __init__(self, model, item_to_idx, idx_to_item, max_session_len=20, device="cpu"):
        self.model = model
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.max_session_len = max_session_len
        self.device = device

    def Recomendar(self, prefix_items, k=10, seen_items=None):
        vistos = set(prefix_items) if seen_items is None else set(seen_items)
        #Si algun item del prefijo no existe en train, simplemente se ignora.
        prefijo_idx = [self.item_to_idx[item] for item in prefix_items if item in self.item_to_idx]
        prefijo_idx = prefijo_idx[-self.max_session_len:]

        if not prefijo_idx:
            return []

        input_ids = torch.tensor([prefijo_idx], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(prefijo_idx)], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, lengths)[0].detach().cpu()

        #Se excluyen padding e items ya vistos antes de tomar el top-k final.
        logits[0] = -torch.inf
        for item in vistos:
            idx = self.item_to_idx.get(item)
            if idx is not None:
                logits[idx] = -torch.inf

        top_k = min(k, len(self.idx_to_item))
        top_indices = torch.topk(logits, k=top_k).indices.tolist()
        return [self.idx_to_item[idx] for idx in top_indices if idx in self.idx_to_item]


def EntrenarGru4Rec(train_sessions, max_session_len=20, embedding_dim=64, hidden_dim=128, dropout=0.2, epochs=2, batch_size=256, learning_rate=1e-3, weight_decay=1e-5, max_train_examples=None, usar_todos_los_prefijos=True, seed=42, device="cpu", verbose=True,):
    #Esta funcion arma todo el flujo de GRU4Rec, desde los datos, modelo, entrenamiento y wrapper final.
    FijarSemilla(seed)
    item_to_idx, idx_to_item = ConstruirMapeoItems(train_sessions)
    ejemplos_train = ConstruirEjemplosSecuenciales(train_sessions, item_to_idx=item_to_idx, max_session_len=max_session_len, usar_todos_los_prefijos=usar_todos_los_prefijos, max_train_examples=max_train_examples, seed=seed,)

    modelo = ModeloGru4Rec(num_items=len(item_to_idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout,)
    history = EntrenarModeloSecuencial(modelo, ejemplos_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, device=device, verbose=verbose,)
    recomendador = RecomendadorSecuencialNeuronal(modelo, item_to_idx, idx_to_item, max_session_len=max_session_len, device=device, )
    return recomendador, history

def EntrenarSasRec(train_sessions, max_session_len=20, embedding_dim=64, num_heads=2, num_layers=2, dropout=0.2, epochs=2, batch_size=256, learning_rate=1e-3, weight_decay=1e-5, max_train_examples=None, usar_todos_los_prefijos=True, seed=42, device="cpu", verbose=True,):
    #Igual que la anterior, pero usando Transformer para la sesion.
    FijarSemilla(seed)
    item_to_idx, idx_to_item = ConstruirMapeoItems(train_sessions)
    ejemplos_train = ConstruirEjemplosSecuenciales(train_sessions, item_to_idx=item_to_idx, max_session_len=max_session_len, usar_todos_los_prefijos=usar_todos_los_prefijos, max_train_examples=max_train_examples, seed=seed,)

    modelo = ModeloSasRec(num_items=len(item_to_idx), max_len=max_session_len, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout,)
    history = EntrenarModeloSecuencial(modelo, ejemplos_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, device=device, verbose=verbose,)
    recomendador = RecomendadorSecuencialNeuronal(modelo, item_to_idx, idx_to_item, max_session_len=max_session_len, device=device,)
    return recomendador, history