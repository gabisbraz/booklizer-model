import torch
from torch.nn import TripletMarginLoss
from torch.optim import Adam
import random


def train_gnn_triplet(model, data, df, epochs=50, lr=1e-3, margin=1.0, device="cpu"):
    """
    Treina o modelo GNN com Triplet Margin Loss.

    Args:
        model: instância de GNN (heterogêneo).
        data: HeteroData com nós e arestas.
        df: DataFrame original (usado para criar pares positivos).
        epochs: número de épocas.
        lr: taxa de aprendizado.
        margin: margem da TripletMarginLoss.
        device: "cpu" ou "cuda".
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = TripletMarginLoss(margin=margin, p=2)

    livros = df["livro"].unique()
    livro2id = {livro: i for i, livro in enumerate(livros)}

    # mapeia livros → gêneros (para positivos)
    livro_generos = {row["livro"]: row["lista_de_generos"] for _, row in df.iterrows()}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward: gera embeddings para cada nó
        embeddings = model(data)
        livro_emb = embeddings["livro"]  # só queremos os embeddings de livros

        # Criar amostras (anchor, positive, negative)
        anchors, positives, negatives = [], [], []
        for livro in livros:
            anchor_id = livro2id[livro]

            # escolher positivo (mesmo gênero ou mesmo autor)
            candidatos_pos = df[
                (df["autor"] == df.loc[df["livro"] == livro, "autor"].values[0])
                | (
                    df["lista_de_generos"].apply(
                        lambda gs: any(g in livro_generos[livro] for g in gs)
                    )
                )
            ]["livro"].tolist()

            candidatos_pos = [p for p in candidatos_pos if p != livro]
            if not candidatos_pos:
                continue  # se não há positivos, pula

            positive_id = livro2id[random.choice(candidatos_pos)]

            # escolher negativo (livro que não compartilha autor nem gênero)
            candidatos_neg = [
                l for l in livros if l != livro and l not in candidatos_pos
            ]
            if not candidatos_neg:
                continue

            negative_id = livro2id[random.choice(candidatos_neg)]

            anchors.append(livro_emb[anchor_id])
            positives.append(livro_emb[positive_id])
            negatives.append(livro_emb[negative_id])

        if not anchors:
            raise ValueError(
                "Não foi possível formar nenhum triplet válido. Verifique o dataset."
            )

        anchors = torch.stack(anchors).to(device)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)

        # Loss
        loss = criterion(anchors, positives, negatives)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model
