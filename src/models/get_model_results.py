import pandas as pd
import plotly.express as px
import torch
from IPython.core.display_functions import display
from sklearn.metrics.pairwise import cosine_similarity


def calcular_similaridade(model, data, df):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        embeddings = model(data.to(device)).cpu().numpy()
    livro_embeddings = embeddings[: len(df)]
    return cosine_similarity(livro_embeddings)


def recomendar_livros(nome_livro, n_recomendacoes, df, similarity_matrix):
    # Verifica se o livro está no DataFrame
    if nome_livro not in df["livro"].values:
        raise ValueError(f"Livro '{nome_livro}' não encontrado no DataFrame.")

    # Índice da linha do livro (não o ID!)
    row_idx = df[df["livro"] == nome_livro].index[0]

    # Obtém as similaridades e ordena
    similaridades = list(enumerate(similarity_matrix[row_idx]))
    similares_ordenados = sorted(similaridades, key=lambda x: x[1], reverse=True)

    # Pega os top-n (ignorando o próprio livro)
    recomendados_idx = [i for i, _ in similares_ordenados if i != row_idx][:n_recomendacoes]

    # Cria o DataFrame de saída
    livros_recomendados = df.iloc[recomendados_idx].copy()
    livros_recomendados["similaridade"] = [similarity_matrix[row_idx][i] for i in recomendados_idx]

    # Adiciona o livro de entrada no topo
    livro_base = df.iloc[[row_idx]].copy()
    livro_base["similaridade"] = 1.0  # similaridade consigo mesmo

    resultado = pd.concat([livro_base, livros_recomendados], ignore_index=True)

    # Concatena os gêneros em string
    resultado["generos"] = resultado["lista_de_generos"].apply(lambda x: ", ".join(x))

    return resultado[["livro", "generos", "descricao", "similaridade"]]


def mostrar_recomendacoes_interativas(
    nome_livro, similarity_matrix, df, embeddings, n_recomendacoes=5
):
    # Garante que o livro existe
    if nome_livro not in df["livro"].values:
        raise ValueError(f"Livro '{nome_livro}' não encontrado.")

    # Gera tabela de recomendações
    resultado = recomendar_livros(
        nome_livro=nome_livro,
        n_recomendacoes=n_recomendacoes,
        df=df,
        similarity_matrix=similarity_matrix,
    )

    display(resultado)

    fig_bar = px.bar(
        resultado[::-1],  # Inverte para mostrar o mais similar em cima
        x="similaridade",
        y="livro",
        color="similaridade",
        color_continuous_scale="Blues",
        orientation="h",
        text="generos",
        title=f"Top {n_recomendacoes} recomendações para: {nome_livro}",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(xaxis_title="Similaridade", yaxis_title="Livro")
    fig_bar.show()

    return resultado