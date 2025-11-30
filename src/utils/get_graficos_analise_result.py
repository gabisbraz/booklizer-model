import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from umap import UMAP

DIR_ROOT = str(Path(__file__).parents[1])
if DIR_ROOT not in sys.path:
    sys.path.append(DIR_ROOT)

from src.utils.get_graficos_analise_dados import salvar_grafico


def plot_loss_umap(losses, embeddings, df, n_neighbors=15, min_dist=0.1):
    """
    losses: lista de floats do TripletLoss por época
    embeddings: tensor ou np.array das embeddings finais (todos os nós)
    df: DataFrame com coluna 'id' e 'lista_de_generos'
    """

    # --- 1. Curva do Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Triplet Loss")
    plt.title("Evolução do Triplet Loss por época")
    plt.grid(True)
    plt.show()

    # --- 2. UMAP ---
    # Pegamos apenas embeddings de livros
    num_livros = len(df)
    livro_embeddings = embeddings[:num_livros]

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2_d = reducer.fit_transform(livro_embeddings)

    # --- 3. Cor pelo gênero principal ---
    # Para simplificação: pega o primeiro gênero de cada livro
    generos_principais = [g[0] if len(g) > 0 else "Desconhecido" for g in df["lista_de_generos"]]
    unique_genres = sorted(set(generos_principais))
    genre_to_color = {g: i for i, g in enumerate(unique_genres)}
    colors = [genre_to_color[g] for g in generos_principais]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding_2_d[:, 0], embedding_2_d[:, 1], c=colors, cmap="tab20", s=60, alpha=0.8
    )
    plt.title("UMAP das embeddings dos livros")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    # legenda
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles,
        unique_genres,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Gênero",
    )
    plt.show()


def plot_umap_interativo(embeddings, df, n_neighbors=15, min_dist=0.1):
    """
    embeddings: np.array ou tensor das embeddings finais (todos os nós)
    df: DataFrame com colunas 'id', 'livro', 'descricao', 'lista_de_generos'
    """
    # Apenas embeddings de livros
    num_livros = len(df)
    livro_embeddings = embeddings[:num_livros]

    # UMAP 2D
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2_d = reducer.fit_transform(livro_embeddings)

    # Gênero principal (primeiro da lista)
    generos_principais = [g[0] if len(g) > 0 else "Desconhecido" for g in df["lista_de_generos"]]

    # Cria DataFrame para Plotly
    df_plot = pd.DataFrame({
        "UMAP1": embedding_2_d[:, 0],
        "UMAP2": embedding_2_d[:, 1],
        "Livro": df["livro"],
        "Descricao": df["descricao"],
        "Genero": generos_principais,
    })

    fig = px.scatter(
        df_plot,
        x="UMAP1",
        y="UMAP2",
        color="Genero",
        hover_data=["Livro", "Descricao"],
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title="UMAP interativo das embeddings dos livros",
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(width=900, height=700)
    fig.show()

    salvar_grafico(fig, "plot_umap_interativo")