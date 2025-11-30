import pandas as pd
from loguru import logger
import torch
import pickle

from src.models.model_train import train_gnn_triplet
from src.models.get_model_results import calcular_similaridade
from src.models.initial_vectorizer import (
    processar_dados,
    data_to_pyg_graph,
    get_metricas_grafo,
    plot_grafo_plotly,
    full_graph_analysis,
    visualizar_subgrafo,
)
from src.models.get_graph import criar_grafo, get_plot_grafo, grafo_para_pyg
from src.utils.format_data import construir_dict_generos
from src.models.gnn_model import GNN
from src.utils.get_metricas import calcular_metricas
from src.utils.get_graficos_analise_result import plot_loss_umap, plot_umap_interativo
from src.utils.get_recomendacao import recomendar_livros, plot_umap_recomendacoes_plotly


def get_model_handler(df: pd.DataFrame):

    # PROCESSA E CRIA O GRAFO
    logger.info("Processando os dados e criando o grafo.")
    data_graph = data_to_pyg_graph(df=df)
    # visualizar_subgrafo(data_graph, num_livros=5)
    # full_graph_analysis(data_graph)
    # get_metricas_grafo(data=data_graph)
    # plot_grafo_plotly(
    #     data=data_graph,
    #     livro_nome=["To Kill a Mockingbird", "Pride and Prejudice", "1984"],
    #     df=df,
    # )

    # CRIA O MODELO GNN
    logger.info("Criando o modelo GNN.")
    model_gnn = GNN(hidden_dim=24, out_dim=64)
    # embeddings = model_gnn.forward(data_graph)

    # TREINA O MODELO
    logger.info("Treinando o modelo.")
    modelo_treinado = train_gnn_triplet(
        model=model_gnn, data=data_graph, df=df, epochs=30, lr=1e-3, margin=1.0
    )

    # Salvar
    with open("gnn_model.pkl", "wb") as f:
        pickle.dump(modelo_treinado.state_dict(), f)

    recomendacoes = recomendar_livros(modelo_treinado, data_graph, "1984", df, top_k=10)
    for livro, score in recomendacoes:
        info = df.loc[df["livro"] == livro][["livro", "autor", "lista_de_generos"]]
        logger.info(
            f"{livro} - Similaridade: {score:.4f} - Generos: {info['lista_de_generos'].values[0]} - Autor: {info['autor'].values[0]}"
        )

    metricas_modelo = calcular_metricas(model=modelo_treinado, data=data_graph, df=df)
    logger.info(f"Métricas do modelo: {metricas_modelo}")
