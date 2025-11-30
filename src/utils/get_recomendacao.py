import sys
from pathlib import Path

import plotly.express as px
import torch
import torch.nn.functional as F
import umap
from torch_geometric.data import HeteroData

DIR_ROOT = str(Path(__file__).parents[1])
if DIR_ROOT not in sys.path:
    sys.path.append(DIR_ROOT)

from src.utils.get_graficos_analise_dados import salvar_grafico


def recomendar_livros(model, data: HeteroData, livro_nome: str, df, top_k=5, device="cpu"):
    """
    Retorna os top-k livros mais similares ao livro_nome usando embeddings do modelo.

    Args:
        model: modelo GNN treinado.
        data: HeteroData com grafo.
        livro_nome: nome do livro de referência.
        df: DataFrame original (para mapear índice ↔ nome do livro).
        top_k: quantidade de recomendações.
        device: "cpu" ou "cuda".
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data)  # dicionário {tipo: tensor}

    livro_emb = embeddings["livro"].to(device)
    livros = df["livro"].unique()
    livro2_id = {l: i for i, l in enumerate(livros)}
    id2_livro = dict(enumerate(livros))

    if livro_nome not in livro2_id:
        raise ValueError(f"Livro '{livro_nome}' não encontrado no dataset.")

    anchor_id = livro2_id[livro_nome]
    anchor_emb = livro_emb[anchor_id].unsqueeze(0)  # shape [1, dim]

    # calcula similaridade de cosseno com todos os livros
    sim = F.cosine_similarity(anchor_emb, livro_emb)  # [num_livros]

    # remove o próprio livro
    sim[anchor_id] = -1.0

    # pega os top-k índices
    topk_ids = torch.topk(sim, k=top_k).indices.tolist()
    topk_livros = [id2_livro[i] for i in topk_ids]
    topk_scores = [sim[i].item() for i in topk_ids]

    return list(zip(topk_livros, topk_scores))


def plot_umap_recomendacoes_plotly(final_embeddings, df_processed, livro_ancora, recomendacoes):
    """
    Plota UMAP interativo (Plotly) dos embeddings dos livros.
    - Todos os livros em cinza
    - Livro âncora em azul
    - Recomendações em vermelho
    """
    reducer = umap.UMAP(random_state=42)
    emb_2_d = reducer.fit_transform(final_embeddings)

    # DataFrame auxiliar para plot
    df_plot = df_processed.copy()
    df_plot["x"] = emb_2_d[df_plot["id_livro"], 0]
    df_plot["y"] = emb_2_d[df_plot["id_livro"], 1]
    df_plot["cor"] = "Outros"

    # Marca livro âncora
    df_plot.loc[df_plot["livro"] == livro_ancora, "cor"] = "Livro Âncora"

    # Marca recomendações
    livros_recs = [livro for livro, _ in recomendacoes]
    df_plot.loc[df_plot["livro"].isin(livros_recs), "cor"] = "Recomendado"

    # Plot interativo
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cor",
        hover_data=["livro", "autor"],
        title=f"UMAP - Recomendações para '{livro_ancora}'",
        color_discrete_map={
            "Outros": "lightgray",
            "Livro Âncora": "blue",
            "Recomendado": "red",
    },)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="black")))
    fig.show()

    salvar_grafico(fig, "plot_umap_recomendacao")