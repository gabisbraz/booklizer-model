import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

DIR_ROOT = str(Path(__file__).parents[1])
if DIR_ROOT not in sys.path:
    sys.path.append(DIR_ROOT)

from src.scripts.model_handler import get_model_handler
from src.utils import limpar_dados


def main():

    # Carrega os dados
    logger.info("Carregando os dados iniciais.")
    df = pd.read_csv("data/01_extract/goodreads_data.csv").head(1000)

    # Limpa os dados
    logger.info("Limpando os dados.")
    df_limpo = limpar_dados(df=df)

    # Cria pastas necessárias
    os.makedirs(
        f"data/02_transform/dataframe_limpo/{datetime.now().strftime("%Y_%m_%d")}",
        exist_ok=True,
    )

    # Salva os dados limpos
    logger.info("Salvando os dados limpos.")
    df_limpo.to_excel(
        f"data/02_transform/dataframe_limpo/{datetime.now().strftime("%Y_%m_%d")}/cleaned_data.xlsx",
        index=False,
    )

    # Executa o model handler
    logger.info("Iniciando o model handler.")
    get_model_handler(df=df_limpo)


if __name__ == "__main__":

    logger.info("Iniciando a aplicação...")
    main()
    logger.info("Aplicação finalizada.")