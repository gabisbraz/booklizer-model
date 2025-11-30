import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import pandas as pd
import random


def calcular_metricas(model, data, df, top_k=10, margin=1.0, device="cpu"):
    """
    Avalia o modelo de recomendação com métricas completas.
    """
    model.eval()
    model = model.to(device)
    livros = df["livro"].unique()
    livro2id = {livro: i for i, livro in enumerate(livros)}

    # TripletMarginLoss
    criterion = TripletMarginLoss(margin=margin, p=2)

    # Mapa de livros para positivos (mesmo autor ou gênero)
    livro_positivos = {}
    livro_generos = {}
    livro_autores = {}
    for _, row in df.iterrows():
        livro = row["livro"]
        autor = row["autor"]
        generos = set(row["lista_de_generos"])
        livro_positivos[livro] = set(
            df[
                (df["autor"] == autor)
                | (df["lista_de_generos"].apply(lambda gs: bool(set(gs) & generos)))
            ]["livro"]
        ) - {livro}
        livro_generos[livro] = generos
        livro_autores[livro] = autor

    # Gera embeddings
    with torch.no_grad():
        embeddings = model(data)
    livro_emb = embeddings["livro"].to(device)

    # Inicializa métricas
    precisions, recalls, f1s, hits, average_precisions, ndcgs = [], [], [], [], [], []
    sim_pos, sim_neg = [], []
    coverage_set = set()
    diversity_list = []

    # Triplet Loss
    triplet_losses = []

    for livro in livros:
        anchor_id = livro2id[livro]
        anchor_emb = livro_emb[anchor_id].unsqueeze(0)

        # Triplets
        positivos = list(livro_positivos[livro])
        negativos = [l for l in livros if l != livro and l not in positivos]
        if positivos and negativos:
            pos_id = livro2id[random.choice(positivos)]
            neg_id = livro2id[random.choice(negativos)]
            t_loss = criterion(
                anchor_emb,
                livro_emb[pos_id].unsqueeze(0),
                livro_emb[neg_id].unsqueeze(0),
            )
            triplet_losses.append(t_loss.item())

        # Similaridade cosseno
        cos_sim = F.cosine_similarity(anchor_emb, livro_emb)
        cos_sim[anchor_id] = -1

        # Top-k recomendados
        topk_vals, topk_idx = torch.topk(cos_sim, k=top_k)
        topk_livros = [livros[i] for i in topk_idx.tolist()]
        coverage_set.update(topk_livros)
        diversity_list.extend([list(livro_generos[l]) for l in topk_livros])

        # Métricas de ranking
        hits_k = sum([1 for l in topk_livros if l in positivos])
        precision = hits_k / top_k
        recall = hits_k / len(positivos) if len(positivos) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        hit = 1.0 if hits_k > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hits.append(hit)

        # MAP@K
        ap = 0.0
        hit_count = 0
        for rank, l in enumerate(topk_livros, start=1):
            if l in positivos:
                hit_count += 1
                ap += hit_count / rank
        ap = ap / min(len(positivos), top_k) if positivos else 0
        average_precisions.append(ap)

        # NDCG@K
        dcg = sum(
            [
                (1 / torch.log2(torch.tensor(rank + 1))) if l in positivos else 0
                for rank, l in enumerate(topk_livros, start=1)
            ]
        )
        idcg = (
            sum(
                [
                    1 / torch.log2(torch.tensor(rank + 1))
                    for rank in range(1, min(len(positivos), top_k) + 1)
                ]
            )
            if positivos
            else 0
        )
        ndcg = (dcg / idcg).item() if idcg > 0 else 0
        ndcgs.append(ndcg)

        # Similaridade positiva/negativa
        pos_sims = [cos_sim[livro2id[l]].item() for l in positivos] if positivos else []
        neg_sims = [
            cos_sim[i].item()
            for i in range(len(livros))
            if livros[i] not in positivos and livros[i] != livro
        ]
        if pos_sims:
            sim_pos.append(sum(pos_sims) / len(pos_sims))
        if neg_sims:
            sim_neg.append(sum(neg_sims) / len(neg_sims))

    # Cobertura e diversidade
    coverage = len(coverage_set) / len(livros)
    all_genres = [g for sublist in diversity_list for g in sublist]
    diversity = len(set(all_genres)) / len(all_genres) if all_genres else 0

    metrics = {
        "mean_precision@k": sum(precisions) / len(precisions),
        "mean_recall@k": sum(recalls) / len(recalls),
        "mean_f1@k": sum(f1s) / len(f1s),
        "hit_rate@k": sum(hits) / len(hits),
        "MAP@k": sum(average_precisions) / len(average_precisions),
        "NDCG@k": sum(ndcgs) / len(ndcgs),
        "mean_cosine_pos": sum(sim_pos) / len(sim_pos) if sim_pos else 0,
        "mean_cosine_neg": sum(sim_neg) / len(sim_neg) if sim_neg else 0,
        "cosine_gap": (
            (sum(sim_pos) / len(sim_pos) - sum(sim_neg) / len(sim_neg))
            if sim_pos and sim_neg
            else 0
        ),
        "triplet_loss_mean": (
            sum(triplet_losses) / len(triplet_losses) if triplet_losses else 0
        ),
        "coverage@k": coverage,
        "diversity@k": diversity,
    }

    return metrics
