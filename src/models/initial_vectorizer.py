import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData


def data_to_pyg_graph(df: pd.DataFrame) -> HeteroData:
    """
    Constrói um grafo heterogêneo PyTorch Geometric a partir de um DataFrame
    com colunas: 'livro' (str), 'autor' (str), 'lista_de_generos' (list[str]).

    - Nós: livros, autores, gêneros
    - Arestas: livro-autor, livro-gênero
    """

    data = HeteroData()

    # Mapear ids únicos para índices de nós
    livros = df["livro"].unique()
    autores = df["autor"].unique()
    generos = sorted({g for lista in df["lista_de_generos"] for g in lista})

    livro2_id = {livro: i for i, livro in enumerate(livros)}
    autor2_id = {autor: i for i, autor in enumerate(autores)}
    genero2_id = {genero: i for i, genero in enumerate(generos)}

    # Criar nós
    data["livro"].num_nodes = len(livros)
    data["autor"].num_nodes = len(autores)
    data["genero"].num_nodes = len(generos)

    # Criar arestas livro-autor
    livro_autor_src = [livro2_id[row["livro"]] for _, row in df.iterrows()]
    livro_autor_dst = [autor2_id[row["autor"]] for _, row in df.iterrows()]

    data["livro", "escrito_por", "autor"].edge_index = torch.tensor(
        [livro_autor_src, livro_autor_dst], dtype=torch.long
    )
    data["autor", "escreveu", "livro"].edge_index = torch.tensor(
        [livro_autor_dst, livro_autor_src], dtype=torch.long
    )  # reverso

    # Criar arestas livro-gênero
    livro_genero_src, livro_genero_dst = [], []
    for row in df.itertuples():
        l_id = livro2_id[row.livro]
        for g in row.lista_de_generos:
            livro_genero_src.append(l_id)
            livro_genero_dst.append(genero2_id[g])

    data["livro", "tem_genero", "genero"].edge_index = torch.tensor(
        [livro_genero_src, livro_genero_dst], dtype=torch.long
    )
    data["genero", "pertence_a", "livro"].edge_index = torch.tensor(
        [livro_genero_dst, livro_genero_src], dtype=torch.long
    )  # reverso

    return data


def visualizar_subgrafo(data, num_livros=5, seed=42):
    """
    Visualiza um subgrafo do HeteroData com alguns livros e seus autores/gêneros.

    Args:
        data (HeteroData): grafo heterogêneo PyG.
        num_livros (int): quantidade de livros para exibir.
        seed (int): semente aleatória para reprodutibilidade.
    """

    random.seed(seed)

    # Extrair nomes dos nós (índices e rótulos)
    livros = list(range(data["livro"].num_nodes))
    autores = list(range(data["autor"].num_nodes))
    generos = list(range(data["genero"].num_nodes))

    # Selecionar livros aleatórios
    livros_sel = random.sample(livros, min(num_livros, len(livros)))

    # Montar conjunto de nós conectados (livros + autores + generos relacionados)
    nos_subgrafo = set(livros_sel)

    # Arestas livro-autor
    src_la, dst_la = data["livro", "escrito_por", "autor"].edge_index.tolist()
    for s, d in zip(src_la, dst_la):
        if s in livros_sel:
            nos_subgrafo.add(d)

    # Arestas livro-genero
    src_lg, dst_lg = data["livro", "tem_genero", "genero"].edge_index.tolist()
    for s, d in zip(src_lg, dst_lg):
        if s in livros_sel:
            nos_subgrafo.add(
                d + data["autor"].num_nodes + data["livro"].num_nodes
            )  # offset para gêneros

    # Converter para NetworkX para visualização
    G = nx.Graph()

    # Adicionar nós com tipo e rótulo
    for i in livros_sel:
        G.add_node(f"livro_{i}", tipo="livro")

    for i in range(data["autor"].num_nodes):
        G.add_node(f"autor_{i}", tipo="autor")

    for i in range(data["genero"].num_nodes):
        G.add_node(f"genero_{i}", tipo="genero")

    # Arestas livro-autor
    for s, d in zip(src_la, dst_la):
        if s in livros_sel:
            G.add_edge(f"livro_{s}", f"autor_{d}")

    # Arestas livro-genero
    for s, d in zip(src_lg, dst_lg):
        if s in livros_sel:
            G.add_edge(f"livro_{s}", f"genero_{d}")

    # Cores por tipo de nó
    cores = {
        "livro": "#1f77b4",  # azul
        "autor": "#2ca02c",  # verde
        "genero": "#ff7f0e",  # laranja
    }
    node_colors = [cores[G.nodes[n]["tipo"]] for n in G.nodes]

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=0.7)

    # Desenhar grafo
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1000,
        font_size=8,
        font_weight="bold",
        edge_color="gray",
    )

    # Legenda
    for tipo, cor in cores.items():
        plt.scatter([], [], c=cor, label=tipo)
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Tipos de Nós")

    plt.title(f"Subgrafo com {num_livros} livros, seus autores e gêneros")
    plt.show()


def get_metricas_grafo(data: HeteroData) -> dict:
    # Número de nós por tipo
    logger.info(f"Nós de livros: {data['livro']['num_nodes']}")
    logger.info(f"Nós de autores: {data['autor']['num_nodes']}")
    logger.info(f"Nós de gêneros: {data['genero']['num_nodes']}")

    # Número de arestas por tipo
    logger.info(
        f"Arestas livro→autor: {data['livro', 'escrito_por', 'autor'].edge_index.shape[1]}",
    )
    logger.info(
        f"Arestas livro→gênero: {data['livro', 'tem_genero', 'genero'].edge_index.shape[1]}",
    )


def plot_grafo_plotly(
    data: HeteroData,
    livro_nome: list[str],
    df: pd.DataFrame,
    output_dir="data/02_transform",
):
    """
    Cria visualização interativa (Plotly) de um ou mais livros e seus nós conectados
    (autores + gêneros), com legenda de tipos de nós, e salva como HTML.
    """

    # Mapas de nomes -> índices
    livros = df["livro"].unique()
    autores = df["autor"].unique()
    generos = sorted({g for lista in df["lista_de_generos"] for g in lista})

    livro2_id = {livro: i for i, livro in enumerate(livros)}
    autor2_id = {autor: i for i, autor in enumerate(autores)}
    genero2_id = {genero: i for i, genero in enumerate(generos)}

    G = nx.Graph()

    # Construir subgrafo apenas dos livros desejados
    for obra in livro_nome:
        if obra not in livro2_id:
            print(f"Aviso: Livro '{obra}' não encontrado, ignorando.")
            continue

        livro_id = livro2_id[obra]
        G.add_node(obra, tipo="livro")

        # Autores conectados
        edge_index = data["livro", "escrito_por", "autor"].edge_index
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if i == livro_id:
                autor_nome = autores[j]
                G.add_node(autor_nome, tipo="autor")
                G.add_edge(obra, autor_nome)

        # Gêneros conectados
        edge_index = data["livro", "tem_genero", "genero"].edge_index
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if i == livro_id:
                genero_nome = generos[j]
                G.add_node(genero_nome, tipo="genero")
                G.add_edge(obra, genero_nome)

    # Layout de posicionamento dos nós
    pos = nx.spring_layout(G, seed=42)

    # Arestas (todas iguais)
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="gray"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Separar nós por tipo (para legenda)
    node_groups = {"livro": [], "autor": [], "genero": []}
    for node, attr in G.nodes(data=True):
        tipo = attr["tipo"]
        x, y = pos[node]
        node_groups[tipo].append((x, y, node))

    # Mapear cores e nomes de legenda
    cores = {
        "livro": ("skyblue", "Livro"),
        "autor": ("lightgreen", "Autor"),
        "genero": ("salmon", "Gênero"),
    }

    node_traces = []
    for tipo, nodes in node_groups.items():
        if not nodes:
            continue
        x_vals = [n[0] for n in nodes]
        y_vals = [n[1] for n in nodes]
        nomes = [n[2] for n in nodes]
        cor, label = cores[tipo]

        node_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=nomes,
                textposition="top center",
                hoverinfo="text",
                marker=dict(size=30, color=cor, line=dict(width=2, color="black")),
                name=label,  # legenda
                showlegend=True,
        ))

    # Combinar arestas + nós
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=f"Conexões do(s) livro(s): {', '.join(livro_nome)}",
            title_x=0.5,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),)

    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{'_'.join(livro_nome).replace(' ', '_')}_viz.html")

    fig.write_html(filepath)
    print(f"✅ Gráfico salvo em: {filepath}")


def processar_dados(data: pd.DataFrame, colunas: list):
    """
    Processa um DataFrame e gera IDs únicos para múltiplas colunas (variáveis) de interesse,
    garantindo que não haja sobreposição de IDs entre elas.

    Parâmetros
    ----------
    data : pd.DataFrame
        DataFrame contendo as colunas de interesse.
    colunas : list
        Lista de colunas a serem processadas. Ex: ['livro', 'autor', 'lista_de_generos'].
        Se a coluna for uma lista (como 'lista_de_generos'), cada item recebe um ID único.

    Retorna
    -------
    data : pd.DataFrame
        DataFrame com novas colunas de ID para cada coluna de entrada.
        Para listas (ex: gêneros), não adiciona coluna no df, mas retorna um dict separadamente.
    encoders : dict
        Dicionário com LabelEncoders para cada coluna (para colunas escalares).
    dict_generos : dict
        Para colunas de listas, retorna dicionário mapeando item -> ID único.
    """
    data = data.reset_index(drop=True).copy()
    encoders = {}
    dict_generos = {}
    offset = 0  # controla o próximo ID disponível

    for col in colunas:
        if data[col].apply(lambda x: isinstance(x, list)).any():
            # coluna de listas (ex: lista_de_generos)
            all_items = [item for sublist in data[col] for item in sublist]
            unique_items = sorted(set(all_items))
            dict_generos[col] = {item: i + offset for i, item in enumerate(unique_items)}
            offset += len(unique_items)

            continue
        le = LabelEncoder()
        data[f"id_{col}"] = le.fit_transform(data[col]) + offset
        offset += len(le.classes_)
        encoders[col] = le

    return data, encoders, dict_generos


def gerar_embeddings_com_autor(
    df: pd.DataFrame, genre_encoder: dict, author_encoder: dict, embedding_dim=64
) -> torch.Tensor:
    """
    Cria embeddings iniciais para livros, gêneros e autores.

    - Livros: embedding do autor + zeros
    - Gêneros: zeros
    - Autores: one-hot ou embedding contínuo simples
    """
    num_books = len(df)
    num_genres = len(genre_encoder)
    num_authors = len(author_encoder)

    num_features = embedding_dim  # dimensão final de cada nó
    all_vectors = torch.zeros(
        (num_books + num_genres + num_authors, num_features), dtype=torch.float32
    )

    # --- Livros ---
    for row in df.itertuples():
        book_id = int(row.id)
        author_id = int(author_encoder[row.autor])
        # Exemplo simples: coloca 1 no índice do autor dentro do vetor do livro
        all_vectors[book_id, author_id % embedding_dim] = 1.0

    # --- Gêneros ---
    for gid in genre_encoder.values():
        all_vectors[gid] = torch.zeros(num_features)  # podem ficar zeros

    # --- Autores ---
    for aid in author_encoder.values():
        all_vectors[aid + num_books + num_genres, aid % embedding_dim] = 1.0  # posição global

    return all_vectors


def full_graph_analysis(data: HeteroData):
    print("===== Análise Exploratória do Grafo Heterogêneo =====\n")

    # 1. Número de nós por tipo
    print("1️⃣ Número de nós por tipo:")
    for node_type in data.node_types:
        print(f"  {node_type}: {data[node_type].num_nodes} nós")

    # 2. Número de arestas por tipo
    print("\n2️⃣ Número de arestas por tipo:")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.size(1)
        print(f"  {edge_type}: {num_edges} arestas")

    # 3. Graus dos nós
    print("\n3️⃣ Estatísticas de grau dos nós por tipo:")
    node_degrees = defaultdict(list)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        for src, dst in edge_index.t().tolist():
            node_degrees[src_type].append(src)
            node_degrees[dst_type].append(dst)

    for node_type in data.node_types:
        degrees = node_degrees[node_type]
        if degrees:
            min_deg = min(degrees.count(i) for i in range(data[node_type].num_nodes))
            max_deg = max(degrees.count(i) for i in range(data[node_type].num_nodes))
            avg_deg = (
                sum(degrees.count(i) for i in range(data[node_type].num_nodes))
                / data[node_type].num_nodes
            )
            print(f"  {node_type}: grau mínimo={min_deg}, máximo={max_deg}, médio={avg_deg:.2f}")
        else:
            print(f"  {node_type}: sem arestas")

    # 4. Nós isolados
    print("\n4️⃣ Nós isolados por tipo:")
    for node_type in data.node_types:
        connected_nodes = set(node_degrees[node_type])
        isolated = set(range(data[node_type].num_nodes)) - connected_nodes
        print(f"  {node_type}: {len(isolated)} nós isolados")

    # 5. Estatísticas gerais
    total_nodes = sum(data[node_type].num_nodes for node_type in data.node_types)
    total_edges = sum(data[edge_type].edge_index.size(1) for edge_type in data.edge_types)
    approx_density = total_edges / (
        total_nodes * (total_nodes - 1)
    )  # simples, para hetero pode ser interpretativa
    print(f"\n5️⃣ Estatísticas gerais:")
    print(f"  Total de nós: {total_nodes}")
    print(f"  Total de arestas: {total_edges}")
    print(f"  Densidade aproximada: {approx_density:.6f}")

    # 6. Exemplos de conexões
    print("\n6️⃣ Exemplos de conexões (primeiras 5 arestas por tipo):")
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        print(f"  {edge_type}: {edge_index[:, :5].tolist()}")


# Uso