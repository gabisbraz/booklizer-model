# BOOKLIZER

## Booklizer - Um aplicativo de recomendação literária personalizada usando GNN​

O presente trabalho descreve o desenvolvimento doBooklizer, um sistema de recomendação literária personalizada baseado em Graph Neural Networks (GNN). O projeto visa propor uma solução inovadora frente às limitações de plataformas de leitura atuais, como o Goodreads, que não contemplam recomendações personalizadas de acordo com os interesses de cada leitor. A principal motivação consiste em proporcionar uma ferramenta moderna, intuitiva e centrada no usuário, capaz de aprimorar a experiência literária por meio de recomendações mais assertivas. A metodologia adotada envolveu o uso do framework Flutter para o desenvolvimento do aplicativo, Dart para o backend e FastAPI para a comunicação com o modelo de recomendação. O modelo foi estruturado sobre um grafo heterogêneo composto por nós que representam livros, gêneros e autores, sendo treinado com camadas HeteroConv e SageConv para capturar as relações e similaridades entre esses elementos. No aplicativo, o usuário interage com livros apresentados individualmente, sinalizando interesse ou desinteresse, o que aciona o modelo para gerar novas recomendações personalizadas. Os resultados obtidos indicamboas recomendações, evidenciando uma solução bastante satisfatória. O Booklizer contribui para o avanço dos sistemas de recomendação literária ao oferecer uma solução diferenciada e personalizada ao perfil de cada leitor.

## Como Executar o Projeto

### Pré-requisitos
- Python 3.7 ou superior instalado no sistema.
- Gerenciador de pacotes pip.

### Instalação das Dependências
1. Clone ou baixe o repositório do projeto.
2. Navegue até o diretório raiz do projeto (onde está localizado o arquivo `requirements.txt`).
3. Execute o comando abaixo para instalar todas as dependências necessárias:

   ```
   pip install -r requirements.txt
   ```

   Este comando instalará as seguintes bibliotecas principais:
   - pandas: Para manipulação de dados.
   - loguru: Para logging estruturado.
   - torch e torch_geometric: Para implementação e treinamento do modelo GNN.
   - matplotlib, plotly e scikit-learn: Para visualizações e métricas.
   - umap-learn: Para redução de dimensionalidade.
   - openpyxl: Para manipulação de arquivos Excel.

### Estrutura dos Dados
O projeto utiliza dados do Goodreads armazenados em `data/01_extract/goodreads_data.csv`. Certifique-se de que este arquivo esteja presente antes de executar o código.

### Execução do Código
1. Abra um terminal ou prompt de comando.
2. Navegue até o diretório raiz do projeto.
3. Execute o script principal com o comando:

   ```
   python src/main.py
   ```

### Fluxo do Código
O código segue o seguinte fluxo de execução:

1. **Carregamento dos Dados**: O script carrega os dados iniciais do arquivo CSV do Goodreads, limitando a 1000 registros para processamento rápido.

2. **Limpeza dos Dados**: Os dados são processados pela função `limpar_dados` do módulo `src.utils`, que remove inconsistências e prepara os dados para análise.

3. **Salvamento dos Dados Limpos**: Os dados limpos são salvos em um arquivo Excel na pasta `data/02_transform/dataframe_limpo/YYYY_MM_DD/cleaned_data.xlsx`, onde `YYYY_MM_DD` representa a data atual.

4. **Processamento do Modelo**:
   - Os dados limpos são convertidos em um grafo PyTorch Geometric (PyG) heterogêneo, representando livros, gêneros e autores como nós.
   - Um modelo GNN é criado com camadas HeteroConv e SageConv para capturar relações complexas.
   - O modelo é treinado usando triplet loss por 30 épocas, com taxa de aprendizado de 0.001 e margem de 1.0.

5. **Salvamento do Modelo**: O estado do modelo treinado é salvo em `data/03_models/YYYY_MM_DD/gnn_model.pkl`.

6. **Geração de Recomendações e Métricas**:
   - Recomendações são geradas para um livro exemplo ("1984"), retornando os top 10 livros similares com pontuações.
   - Métricas de avaliação do modelo são calculadas e exibidas no log.

Durante a execução, logs detalhados são exibidos no console, informando o progresso de cada etapa. O modelo gera recomendações personalizadas baseadas em similaridades aprendidas no grafo.

### Observações
- O treinamento do modelo pode levar alguns minutos, dependendo do hardware.
- Certifique-se de que há espaço suficiente em disco para os arquivos de saída.
- Para personalizar o treinamento (ex.: número de épocas, taxa de aprendizado), edite os parâmetros no arquivo `src/scripts/model_handler.py`.
