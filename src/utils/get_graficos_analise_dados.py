import os
import pandas as pd
import ast
from datetime import date
from loguru import logger
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

from .get_limpeza_dados import limpar_dados


# ==============================
# FUNÇÃO: SALVAR GRÁFICOS
# ==============================
def salvar_grafico(fig, nome: str):
    pasta = f"data/05_analises/graficos_{date.today().strftime('%d_%m')}"
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{nome}.html")
    fig.write_html(caminho, include_plotlyjs="cdn")
    logger.info(f"Gráfico salvo: {caminho}")


# ==============================
# FUNÇÃO: GÊNEROS MAIS POPULARES
# ==============================
def grafico_generos(df: pd.DataFrame):
    df_exploded = df.explode("Genres").reset_index(drop=True)
    df_generos_count = (
        df_exploded.groupby("Genres")["Book"]
        .count()
        .reset_index()
        .rename(columns={"Book": "Qtd_Livros"})
        .sort_values(by="Qtd_Livros", ascending=False)
    )

    fig = px.bar(
        df_generos_count.head(25),
        x="Qtd_Livros",
        y="Genres",
        orientation="h",
        title="Top 25 Gêneros com Mais Livros",
        labels={"Qtd_Livros": "Quantidade de Livros", "Genres": "Gênero"},
        color="Qtd_Livros",
        color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    salvar_grafico(fig, "top_generos")


# ==============================
# FUNÇÃO: AUTORES MAIS PRODUTIVOS
# ==============================
def grafico_autores(df: pd.DataFrame):
    df_author_count = (
        df.groupby("Author")["Book"]
        .count()
        .reset_index()
        .rename(columns={"Book": "Qtd_Livros"})
        .sort_values(by="Qtd_Livros", ascending=False)
    )

    fig = px.bar(
        df_author_count.head(20),
        x="Qtd_Livros",
        y="Author",
        orientation="h",
        title="Top 20 Autores com Mais Livros",
        labels={"Qtd_Livros": "Quantidade de Livros", "Author": "Autor"},
        color="Qtd_Livros",
        color_continuous_scale="Sunsetdark",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    salvar_grafico(fig, "top_autores")


# ==============================
# FUNÇÃO: PALAVRAS MAIS FREQUENTES NAS DESCRIÇÕES
# ==============================
def grafico_palavras(df: pd.DataFrame):
    descriptions = df["Description"].dropna().tolist()
    vectorizer = CountVectorizer(stop_words="english", max_features=30)
    X = vectorizer.fit_transform(descriptions)

    word_freq = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    df_words = pd.DataFrame({"Palavra": words, "Frequência": word_freq})
    df_words = df_words.sort_values(by="Frequência", ascending=True)

    fig = px.bar(
        df_words,
        x="Frequência",
        y="Palavra",
        orientation="h",
        title="Palavras Mais Frequentes nas Descrições",
        color="Frequência",
        color_continuous_scale="Plasma",
    )
    salvar_grafico(fig, "palavras_frequentes")


# ==============================
# FUNÇÃO: GÊNERO PREDOMINANTE POR AUTOR
# ==============================
def grafico_genero_predominante(df: pd.DataFrame):
    df_exploded = df.explode("Genres").reset_index(drop=True)
    df_author_genre = (
        df_exploded.groupby(["Author", "Genres"])
        .agg(Num_Livros=("Book", "count"))
        .reset_index()
    )

    df_author_genre["max_por_autor"] = df_author_genre.groupby("Author")[
        "Num_Livros"
    ].transform("max")
    df_predominante = df_author_genre[
        df_author_genre["Num_Livros"] == df_author_genre["max_por_autor"]
    ]

    top_autores = (
        df.groupby("Author")["Book"].count().sort_values(ascending=False).head(30).index
    )
    df_predominante_top = df_predominante[df_predominante["Author"].isin(top_autores)]

    # Treemap
    fig1 = px.treemap(
        df_predominante_top,
        path=["Genres", "Author"],
        values="Num_Livros",
        title="Associação entre Autor e Gênero Predominante (Top 30 Autores)",
        color="Genres",
    )
    salvar_grafico(fig1, "genero_predominante_treemap")

    # Barplot
    fig2 = px.bar(
        df_predominante_top,
        x="Num_Livros",
        y="Author",
        color="Genres",
        orientation="h",
        title="Gênero Predominante dos Top 30 Autores",
        labels={
            "Num_Livros": "Número de Livros",
            "Author": "Autor",
            "Genres": "Gênero",
        },
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    salvar_grafico(fig2, "genero_predominante_bar")


def get_graficos_analise_dados(df: pd.DataFrame):
    grafico_generos(df)
    grafico_autores(df)
    grafico_palavras(df)
    grafico_genero_predominante(df)
