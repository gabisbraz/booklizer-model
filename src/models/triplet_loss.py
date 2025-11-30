import numpy as np

import random


def gerar_triplets_batch_hard(embeddings, df, genre_encoder):
    """
    Gera triplets usando o método "batch hard":
    - para cada livro (anchor), escolhe:
        - o gênero que está mais distante (positive)
        - o gênero que está mais próximo, mas que NÃO é do livro (negative)
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    num_livros = len(df)
    triplets = []
    # Cria um mapeamento id -> lista de gêneros (usando a coluna 'id' explicitamente)
    id_to_generos = {
        int(row["id_livro"]): set(row["lista_de_generos"]) for _, row in df.iterrows()
    }

    for book_id in range(num_livros):
        anchor_emb = embeddings_np[book_id]
        anchor_generos = id_to_generos.get(book_id, set())

        id_generos = {g: genre_encoder[g] for g in anchor_generos if g in genre_encoder}
        pos_dists = []
        for genero, idx in id_generos.items():
            dist = np.linalg.norm(anchor_emb - embeddings_np[idx])
            pos_dists.append((idx, dist))
        if not pos_dists:
            continue
        positive = max(pos_dists, key=lambda x: x[1])[0]

        generos_nao_pertencem = set(genre_encoder.keys()) - anchor_generos
        neg_dists = []
        for genero in generos_nao_pertencem:
            idx = genre_encoder[genero]
            dist = np.linalg.norm(anchor_emb - embeddings_np[idx])
            neg_dists.append((idx, dist))
        if not neg_dists:
            continue
        negative = min(neg_dists, key=lambda x: x[1])[0]
        triplets.append((book_id, positive, negative))
    return triplets


def gerar_triplets_hard(edge_index, num_nodes, generos_por_id, num_triplets=2048):
    """
    Um triplet tem três partes:

    - Anchor (âncora): um nó (livro ou gênero).
    - Positive (positivo): um nó que está ligado à âncora (por exemplo, um gênero que o livro tem).
    - Negative (negativo): um nó que não está ligado à âncora e não compartilha gêneros com ela.

    Args:
        edge_index: conexões do grafo (arestas), em forma de tensor.
        num_nodes: número total de nós (livros + gêneros).
        generos_por_id: um dicionário que dá os gêneros de cada nó (quando é livro).
        num_triplets: quantos triplets criar.

        Returns:
            _type_: _description_
    """

    # TRANSFORMA AS ARESTAS EM LISTA DE TUPLAS
    edge_set = set(map(tuple, edge_index.t().tolist()))

    # NÓS SAEM, NÓS VÃO CHEGAR
    src, dst = edge_index

    # VAI ARMAZENAR OS TRIPLETS
    triplets = []

    # PARA CADA TRIPLET
    for _ in range(num_triplets):

        # ESCOLHE ARESTA ALEATÓRIA PARA NÃO GERAR VIÉS
        i = random.randint(0, len(src) - 1)
        anchor = src[i].item()  # Nó âncora (livro)
        positive = dst[i].item()  # Nó positivo (gênero ligado ao livro)

        anchor_generos = generos_por_id.get(anchor, set())  # Gêneros do livro âncora

        """Para todo nó n na rede, ele só considera negativos que: 
            - Não estejam ligados diretamente ao anchor.
            - Não tenham gêneros em comum com o anchor (o & é a interseção de conjuntos).
            Isso garante que o negativo seja um nó bem diferente do anchor (difícil de confundir).
            **diferentes o bastante do anchor para ajudar o modelo a aprender a separar coisas distintas."""
        hard_negatives = [
            n
            for n in range(num_nodes)
            if (anchor, n) not in edge_set
            and len(anchor_generos & generos_por_id.get(n, set())) < 2
        ]

        if not hard_negatives:
            continue

        negative = random.choice(hard_negatives)

        # ADICIONA NA LISTA
        triplets.append((anchor, positive, negative))

    return triplets
