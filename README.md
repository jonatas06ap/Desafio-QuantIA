# Desafio-QuantIA

# 📈 Crypto AI Portfolio Manager

Um assistente inteligente de código aberto para gerenciamento, análise e otimização de carteiras de criptomoedas, utilizando Python e modelos de Inteligência Artificial.

## 📜 Sobre o Projeto

O **Crypto AI Portfolio Manager** é uma ferramenta desenvolvida para investidores de criptomoedas que desejam ir além das planilhas e aplicar ciência de dados e IA na gestão de seus ativos. O projeto conecta-se a exchanges, consolida seu portfólio, realiza análises de risco e utiliza modelos de machine learning para sugerir otimizações e prever tendências.

## ✨ Principais Funcionalidades

  * **Conexão Multi-Exchange:** Agregue seu portfólio de diversas corretoras (Binance, Coinbase, etc.) usando APIs.
  * **Dashboard de Performance:** Visualize o valor total da sua carteira, P\&L (Lucros e Perdas), alocação de ativos e histórico de performance.
  * **Análise de Risco:** Calcule métricas essenciais como Volatilidade, Índice de Sharpe, VaR (Value at Risk) e correlação entre ativos.
  * **Otimização de Carteira (IA):** Utilize a Teoria Moderna do Portfólio (Markowitz) para encontrar a "Fronteira Eficiente" e sugerir a alocação ideal para maximizar o retorno ajustado ao risco.
  * **Previsão de Preços (IA):** Modelos de séries temporais (como LSTM ou Prophet) para prever tendências de preços de curto prazo.
  * **Análise de Sentimento (IA):** Monitore o sentimento do mercado (Fear & Greed) com base em notícias e mídias sociais (Twitter) usando NLP.

## 🤖 O Papel da Inteligência Artificial

Este projeto utiliza IA em três frentes principais:

1.  **Otimização de Portfólio:**

      * **Técnica:** Otimização de Média-Variância (Markowitz) e algoritmos genéticos.
      * **Objetivo:** Encontrar a alocação de ativos que oferece o maior retorno esperado para um determinado nível de risco (ou o menor risco para um determinado retorno).

2.  **Previsão de Séries Temporais:**

      * **Técnica:** Modelos de Deep Learning (Redes Neurais Recorrentes, especificamente LSTMs) e modelos estatísticos (ARIMA, Prophet).
      * **Objetivo:** Analisar dados históricos de preços (OHLCV) para identificar padrões e prever movimentos futuros de preços.

3.  **Análise de Sentimento (NLP):**

      * **Técnica:** Processamento de Linguagem Natural (NLP) com bibliotecas como NLTK ou Transformers (Hugging Face).
      * **Objetivo:** Classificar o sentimento de notícias e postagens em redes sociais como positivo, negativo ou neutro, gerando um indicador de "humor" do mercado.

## 🛠️ Tecnologias Utilizadas

  * **Core:** Python 3.9+
  * **Análise de Dados:** Pandas, NumPy, SciPy
  * **Machine Learning / IA:** Scikit-learn, TensorFlow / Keras, PyTorch
  * **Séries Temporais:** Prophet, Pmdarima
  * **Conexão com Exchanges:** `ccxt`
  * **Web Framework / Dashboard:** Streamlit (preferencial) ou Flask / Plotly Dash
  * **Visualização de Dados:** Plotly, Matplotlib, Seaborn
  * **Banco de Dados:** SQLite (padrão) ou PostgreSQL (opcional)

## 🚀 Instalação e Configuração

Siga estes passos para configurar o ambiente de desenvolvimento.

### 1\. Pré-requisitos

  * Python 3.9 ou superior
  * Git

### 2\. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/crypto-ai-portfolio.git
cd crypto-ai-portfolio
```

### 3\. Crie um Ambiente Virtual

É altamente recomendado usar um ambiente virtual (venv) para isolar as dependências do projeto.

```bash
# Para Unix/macOS
python3 -m venv venv
source venv/bin/activate

# Para Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4\. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 5\. Configure suas Chaves de API

Para se conectar às exchanges, você precisará de chaves de API.

1.  Copie o arquivo de exemplo:

    ```bash
    cp .env.example .env
    ```

2.  Edite o arquivo `.env` e adicione suas chaves de API.
    **NUNCA** envie seu arquivo `.env` com chaves reais para o GitHub. Ele já está incluído no `.gitignore`.

    ```ini
    # .env
    BINANCE_API_KEY=sua_api_key_aqui
    BINANCE_API_SECRET=seu_api_secret_aqui

    # Chave para API do Twitter (para Análise de Sentimento)
    TWITTER_BEARER_TOKEN=seu_bearer_token_aqui
    ```

## 🏃 Como Usar

Para iniciar a aplicação (assumindo o uso de Streamlit):

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no seu navegador para ver o dashboard.

-----

### Exemplo de Análise (Dashboard)

*(Aqui você pode adicionar screenshots do seu projeto quando estiverem prontos)*

`[Screenshot do Dashboard Principal]`
*Descrição: Dashboard principal mostrando o valor total do portfólio e a alocação de ativos.*

`[Screenshot da Fronteira Eficiente]`
*Descrição: Gráfico da Fronteira Eficiente gerado pela IA, mostrando a alocação de risco vs. retorno.*

-----

## 🗺️ Roadmap (Futuras Implementações)

  * [ ] Integrar mais exchanges (Kraken, KuCoin).
  * [ ] Implementar um módulo de backtesting robusto para as estratégias de IA.
  * [ ] Adicionar alertas (Telegram/Email) para mudanças significativas no portfólio ou sinais de IA.
  * [ ] Melhorar os modelos de NLP para uma análise de sentimento mais granular.
  * [ ] "Dockerizar" a aplicação para facilitar o deploy.

## 🤝 Como Contribuir

Contribuições são muito bem-vindas\! Se você tem ideias para melhorias ou encontrou um bug, sinta-se à vontade para:

1.  Fazer um **Fork** do projeto.
2.  Criar uma nova **Branch** (`git checkout -b feature/sua-feature`).
3.  Fazer o **Commit** das suas mudanças (`git commit -m 'Adiciona nova feature'`).
4.  Fazer o **Push** para a Branch (`git push origin feature/sua-feature`).
5.  Abrir um **Pull Request**.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](https://www.google.com/search?q=LICENSE) para mais detalhes.

