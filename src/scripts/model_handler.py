import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from src.models.gnn_model import GNN

DIR_ROOT = str(Path(__file__).parents[1])
if DIR_ROOT not in sys.path:
    sys.path.append(DIR_ROOT)

from src.models.initial_vectorizer import data_to_pyg_graph
from src.models.model_train import train_gnn_triplet
from src.utils.get_metricas import calcular_metricas
from src.utils.get_recomendacao import recomendar_livros


def get_model_handler(df: pd.DataFrame):

    # PROCESSA E CRIA O GRAFO
    logger.info("Processando os dados e criando o grafo.")
    data_graph = data_to_pyg_graph(df=df)

    # CRIA O MODELO GNN
    logger.info("Criando o modelo GNN.")
    model_gnn = GNN(hidden_dim=24, out_dim=64)

    # TREINA O MODELO
    logger.info("Treinando o modelo.")
    modelo_treinado = train_gnn_triplet(
        model=model_gnn, data=data_graph, df=df, epochs=30, lr=1e-3, margin=1.0
    )

    # Cria pastas necessárias
    os.makedirs(
        f"data/03_models/{datetime.now().strftime("%Y_%m_%d")}",
        exist_ok=True,
    )

    # SALVA O MODELO TREINADO
    logger.info("Salvando o modelo treinado.")
    with open(f"data/03_models/{datetime.now().strftime("%Y_%m_%d")}/gnn_model.pkl", "wb") as f:
        pickle.dump(modelo_treinado.state_dict(), f)

    # GERA RECOMENDAÇÕES E CALCULA MÉTRICAS
    logger.info("Gerando recomendações e calculando métricas.")
    recomendacoes = recomendar_livros(modelo_treinado, data_graph, "1984", df, top_k=10)
    for livro, score in recomendacoes:
        info = df.loc[df["livro"] == livro][["livro", "autor", "lista_de_generos"]]
        logger.info(
            '{} - Similaridade: {} - Generos: {} - Autor: {}',
            livro,
            score,
            info["lista_de_generos"].values[0],
            info["autor"].values[0],
        )

    metricas_modelo = calcular_metricas(model=modelo_treinado, data=data_graph, df=df)
    logger.info("Métricas do modelo: {}", metricas_modelo)