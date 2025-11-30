import ast

import pandas as pd
from loguru import logger


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:

    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df = df.loc[df["Author"] != "Anonymous"]

    # Remover duplicados
    df.drop_duplicates(subset=["Book"], inplace=True)

    # Converter string de gêneros em lista
    df["Genres"] = df["Genres"].apply(ast.literal_eval)

    # Manter apenas livros com mais de 1 gênero
    df = df[df["Genres"].apply(lambda x: len(x) >= 1 if isinstance(x, list) else False)]

    # Contar gêneros únicos
    df_exploded = df.explode("Genres").reset_index(drop=True)
    num_generos = df_exploded["Genres"].nunique()

    df.rename(
        columns={
            "Genres": "lista_de_generos",
            "Book": "livro",
            "Author": "autor",
            "Description": "descricao",
        },
        inplace=True,
    )

    logger.info(
        "Base carregada com {} livros únicos, {} autores e {} gêneros distintos.",
        len(df),
        df["autor"].nunique(),
        num_generos,
    )

    return df