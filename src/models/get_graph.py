import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import torch
from torch_geometric.utils import from_networkx


def criar_grafo(data, dict_generos, colunas):
    g = nx.Graph()  # grafo não direcionado

    # Adiciona nós para colunas escalares e listas
    for col in colunas:
        if col in dict_generos:
            # coluna de listas
            for item, id_val in dict_generos[col].items():
                g.add_node(id_val, label=item, tipo=col)
        else:
            # coluna escalar
            for idx, id_val in enumerate(data[f"id_{col}"]):
                g.add_node(id_val, label=data[col].iloc[idx], tipo=col)

    # Adiciona arestas
    for idx, row in data.iterrows():
        for col in colunas:
            if col in dict_generos:
                # lista de itens
                for item in row[col]:
                    g.add_edge(row["id_livro"], dict_generos[col][item])
            else:
                # escalar diferente de livro (assumindo "livro" como principal)
                if col != "livro":
                    g.add_edge(row["id_livro"], row[f"id_{col}"])

    return g


def grafo_para_pyg(grafo, embedding_dim=16, usar_onehot=True):
    """
    Converte um grafo NetworkX em objeto PyTorch Geometric (Data),
    adicionando embeddings iniciais para cada nó.

    Parâmetros
    ----------
    grafo : networkx.Graph
        Grafo criado (por exemplo, a partir de livros, autores, gêneros).
    embedding_dim : int, opcional (default=16)
        Dimensão do vetor de características de cada nó.
    usar_onehot : bool, opcional (default=True)
        Se True, usa one-hot encoding para representar nós.
        Se False, gera embeddings aleatórios.

    Retorna
    -------
    data_pyg : torch_geometric.data.Data
        Objeto compatível com PyTorch Geometric contendo:
        - x: tensor [num_nos, embedding_dim] com features dos nós.
        - edge_index: tensor [2, num_arestas] com conexões.
    """

    num_nos = len(grafo.nodes)

    if usar_onehot:
        # One-hot (identidade): cada nó é uma base diferente
        x = torch.eye(num_nos)
    else:
        # Vetores aleatórios
        x = torch.randn((num_nos, embedding_dim))

    # Adiciona feature 'x' em cada nó
    for i, node in enumerate(grafo.nodes):
        grafo.nodes[node]["x"] = x[i]

    # Converte NetworkX -> PyTorch Geometric
    data_pyg = from_networkx(grafo)

    # Garante que x está no formato tensor
    data_pyg.x = torch.stack([grafo.nodes[n]["x"] for n in grafo.nodes], dim=0)

    return data_pyg


def get_plot_grafo(grafo):
    # Pega labels e tipos
    labels = nx.get_node_attributes(grafo, "label")
    tipos = nx.get_node_attributes(grafo, "tipo")

    # Define cores por tipo de coluna
    color_map = {
        "livro": "lightgreen",
        "autor": "skyblue",
        "lista_de_generos": "salmon",
    }
    node_colors = [color_map.get(tipos[n], "grey") for n in grafo.nodes()]

    # Layout dos nós
    pos = nx.spring_layout(grafo, seed=42)

    # Extrai coordenadas dos nós
    x_nodes = [pos[n][0] for n in grafo.nodes()]
    y_nodes = [pos[n][1] for n in grafo.nodes()]

    # Extrai coordenadas das arestas
    edge_x = []
    edge_y = []
    for edge in grafo.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Cria traço das arestas
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Cria traço dos nós
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=[labels[n] for n in grafo.nodes()],
        textposition="top center",
        hoverinfo="text",
        marker=dict(color=node_colors, size=20, line_width=2),
    )

    # Cria figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),)

    # Salva como HTML
    pio.write_html(fig, file="plot_grafo.html", auto_open=True)