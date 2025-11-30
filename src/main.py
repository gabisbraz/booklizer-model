from loguru import logger
import pandas as pd
import sys
from pathlib import Path
import os
from datetime import datetime

DIR_ROOT = str(Path(__file__).parents[1])
if DIR_ROOT not in sys.path:
    sys.path.append(DIR_ROOT)


from src.utils import limpar_dados, get_graficos_analise_dados
from src.scripts.model_handler import get_model_handler


def main():

    # Cria pastas necessárias
    os.makedirs(
        f"data/02_transform/dataframe_limpo/{datetime.now().strftime("dd_mm_yyyy")}",
        exist_ok=True,
    )

    # Carrega os dados
    logger.info("Carregando os dados iniciais.")
    df = pd.read_csv("data/01_extract/goodreads_data.csv").head(1000)

    # Limpa os dados
    logger.info("Limpando os dados.")
    df_limpo = limpar_dados(df=df)

    # Gera gráficos de análise dos dados
    # logger.info("Gerando gráficos de análise dos dados.")
    # get_graficos_analise_dados(df=df_limpo)

    # Salva os dados limpos
    logger.info("Salvando os dados limpos.")
    df_limpo.to_excel(
        f"data/02_transform/dataframe_limpo/{datetime.now().strftime("dd_mm_yyyy")}/cleaned_data.xlsx",
        index=False,
    )

    # Executa o model handler
    logger.info("Iniciando o model handler.")
    get_model_handler(df=df_limpo)


if __name__ == "__main__":

    logger.info("Iniciando a aplicação...")
    main()
    logger.info("Aplicação finalizada.")
