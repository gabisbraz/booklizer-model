def construir_dict_generos(df, genre_encoder):
    """
    Ela cria um dicionário onde:
        - A chave é o id de cada livro (um número único que representa cada livro).
        - O valor é um conjunto (set) com os gêneros daquele livro, mas só os gêneros que estão no genre_encoder.
    """
    generos_por_id = {}
    for row in df.itertuples():
        book_id = row.id
        generos_por_id[book_id] = {g for g in row.lista_de_generos if g in genre_encoder}
    return generos_por_id