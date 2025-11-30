import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def gerar_graficos_e_tabelas(csv_file="relatorio_final_tp2.csv"):
    """
    Lê o CSV gerado e cria gráficos de análise.
    """
    # 1. Verificação de segurança (Impede o erro FileNotFoundError)
    if not os.path.exists(csv_file):
        print(f"\n[ERRO] O arquivo '{csv_file}' não foi encontrado!")
        print("-> Por favor, execute a OPÇÃO 2 (Rodar Experimentos) primeiro para gerar os dados.")
        return

    print(f"Lendo dados de {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Erro ao abrir o CSV: {e}")
        return
    
    # Configuração visual
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11})

    # Filtrar erros (-1 na silhueta)
    if 'Silhueta' in df.columns:
        df = df[df['Silhueta'] > -1]

    # --- GRÁFICO 1: Impacto da Elongação ---
    print("Gerando 'grafico_elongacao.png'...")
    try:
        elon_df = df[df['Dataset'].str.contains("Syn_Elon", na=False)].copy()
        if not elon_df.empty:
            # Correção do SyntaxWarning: Usando r'' (raw string)
            elon_df.loc[:, 'Nivel'] = elon_df['Dataset'].str.extract(r'(\d+)').astype(int)
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=elon_df[elon_df['Algoritmo'] == 'Refinement'], 
                x='Nivel', y='ARI', hue='Distancia', style='Distancia', 
                markers=True, linewidth=2
            )
            plt.title("Impacto da Elongação na Qualidade (ARI)")
            plt.ylabel("Adjusted Rand Index (ARI)")
            plt.xlabel("Nível de Elongação")
            plt.tight_layout()
            plt.savefig("grafico_elongacao.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Aviso: erro no gráfico de elongação ({e})")

    # --- GRÁFICO 2: Comparação de Raio (UCI) ---
    print("Gerando 'grafico_raio_uci.png'...")
    try:
        uci_df = df[(df['Dataset'].str.contains("UCI", na=False)) & (df['Distancia'] == 'Euclidiana')].copy()
        
        if not uci_df.empty:
            rows = []
            for ds in uci_df['Dataset'].unique():
                sub = uci_df[uci_df['Dataset'] == ds]
                try:
                    # Normaliza pelo raio do Gonzalez
                    base_vals = sub[sub['Algoritmo'] == 'Gonzalez']['Raio_Mean'].values
                    if len(base_vals) > 0:
                        base = base_vals[0]
                        if base > 0:
                            for _, row in sub.iterrows():
                                new_row = row.copy()
                                new_row['Raio_Norm'] = row['Raio_Mean'] / base
                                rows.append(new_row)
                except:
                    pass 

            if rows:
                df_norm = pd.DataFrame(rows)
                # Filtrar para limpar o gráfico
                df_norm = df_norm[df_norm['Parametro'].isin(['15_Runs_Avg', 'Width_1%'])]
                
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_norm, x='Dataset', y='Raio_Norm', hue='Algoritmo', palette="viridis")
                plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
                plt.title("Raio Normalizado (Gonzalez = 1.0)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig("grafico_raio_uci.png", dpi=300)
                plt.close()
    except Exception as e:
        print(f"Aviso: erro no gráfico de raio ({e})")

    # --- TABELA RESUMO ---
    print("\n--- TABELA RESUMO (Médias por Algoritmo nos Reais) ---")
    try:
        uci_only = df[df['Dataset'].str.contains("UCI", na=False)]
        summary = uci_only.groupby(['Algoritmo', 'Distancia'])[['Raio_Mean', 'Tempo_Mean', 'ARI']].mean()
        print(summary)
        summary.to_csv("tabela_resumo_uci.csv")
        print("\n(Tabela salva em 'tabela_resumo_uci.csv')")
    except Exception as e:
        print(f"Erro na tabela: {e}")

    print("\nProcesso concluído! Verifique os arquivos PNG gerados.")