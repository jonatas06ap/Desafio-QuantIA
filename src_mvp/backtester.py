# Em src_mvp/backtester.py

import pandas as pd
import joblib
import vectorbt as vbt
import argparse
import sys
import os

try:
    from feature_builder import load_data_from_files
except ImportError:
    print("Erro: 'feature_builder.py' não encontrado.", file=sys.stderr)
    sys.exit(1)

# COLE ISTO (logo abaixo de 'import os'):

def split_data_chronological(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Divide os dados de séries temporais em treino e teste cronologicamente.
    """
    print(f"Dividindo dados cronologicamente (Test size: {test_size})...")
    
    # Calcula o índice de divisão
    split_index = int(len(X) * (1 - test_size))
    
    # Divide os dataframes
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"  {len(X_train)} amostras de treino (de {X_train.index.min()} a {X_train.index.max()})")
    print(f"  {len(X_test)} amostras de teste (de {X_test.index.min()} a {X_test.index.max()})")
    
    return X_train, X_test, y_train, y_test

def run_backtest_pipeline(model_path: str, 
                          feature_data_path: str, 
                          ohlcv_data_path: str, 
                          news_data_path: str, 
                          granularity: str = 'D',
                          # --- NOVO PARÂMETRO ---
                          run_on_test_set_only: bool = True):
    """
    Orquestra o pipeline de backtesting.
    Se 'run_on_test_set_only' for True, executa a simulação APENAS
    no conjunto de teste (os 20% finais) para validar o overfitting.
    """
    
    print("Iniciando pipeline de backtesting...")
    
    # 1. Carregar Modelo
    model = joblib.load(model_path)

    # 2. Carregar Features
    df_features = pd.read_csv(feature_data_path, index_col=0, parse_dates=True)

    # 3. Carregar Dados OHLCV Originais
    df_ohlcv, _ = load_data_from_files(ohlcv_data_path, news_data_path)
    prices = df_ohlcv[['open', 'close']]

    # 4. Alinhar Dados
    print("Alinhando features e dados de preço...")
    df_full_data = df_features.join(prices, how='inner').dropna()

    # 5. Preparar dados (X e y)
    cols_to_ignore = ['target', 'sentiment_mean', 'news_volume', 'open', 'close']
    features = [col for col in df_full_data.columns if col not in cols_to_ignore]
    
    X_full = df_full_data[features]
    y_full = df_full_data['target'] # Precisamos do 'y' para o split
    
    # 6. Dividir os dados (ou não)
    if run_on_test_set_only:
        print("Executando backtest APENAS no conjunto de teste (20% finais)...")
        # Usa a MESMA função de split do 'model_trainer' para garantir consistência
        _, X_to_predict, _, _ = split_data_chronological(X_full, y_full, test_size=0.2)
        
        # Alinha os dados de preço 'open' com o conjunto de teste X
        open_price_data = df_full_data['open'].reindex(X_to_predict.index)
        
    else:
        print("Executando backtest no conjunto de dados COMPLETO (Treino + Teste)...")
        X_to_predict = X_full
        open_price_data = df_full_data['open']

    # 7. Gerar Sinais
    print("Gerando sinais (predições) do modelo...")
    sinais = model.predict(X_to_predict)
    signals_s = pd.Series(sinais, index=X_to_predict.index, name="Signal")
    
    # 8. Definir Entradas e Saídas (COM ATRASO)
    entries = (signals_s == 1).shift(1).fillna(False)
    exits = (signals_s == 0).shift(1).fillna(False)
    
    # 9. Executar a Simulação
    print(f"Executando simulação (Frequência: {granularity}) usando preço 'open'...")
    pf = vbt.Portfolio.from_signals(
        open_price_data,
        entries,
        exits,
        freq=granularity
    )
    
    print("Backtest concluído.")
    return pf

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Backtesting (Fase 5)")
    
    parser.add_argument("model_path", help="Caminho para o 'trained_model.pkl'.")
    parser.add_argument("feature_data_path", help="Caminho para o 'dataset_treino_final.csv'.")
    parser.add_argument("ohlcv_data_path", help="Caminho para o CSV OHLCV original.")
    parser.add_argument("news_data_path", help="Caminho para o JSON de Notícias original (necessário para alinhamento).")
    parser.add_argument("--granularity", default="D", help="Frequência dos dados (ex: 'D', 'H').")
    parser.add_argument("--stats_out", default="backtest_stats.csv", help="Caminho para salvar as estatísticas (CSV).")
    
    args = parser.parse_args()
    
    try:
        # Executa o pipeline
        portfolio = run_backtest_pipeline(
            model_path=args.model_path,
            feature_data_path=args.feature_data_path,
            ohlcv_data_path=args.ohlcv_data_path,
            news_data_path=args.news_data_path,
            granularity=args.granularity
        )
        
        # Salva as estatísticas
        stats = portfolio.stats()
        stats.to_csv(args.stats_out)
        
        print(f"\nSucesso! Estatísticas do backtest salvas em: {args.stats_out}")
        print("\n--- Resumo das Métricas ---")
        print(stats)
        
    except Exception as e:
        print(f"\nOcorreu um erro durante o backtest: {e}", file=sys.stderr)
