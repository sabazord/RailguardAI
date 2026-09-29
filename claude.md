\# RailGuard AI — Contexto para Claude Code



\## O que é esse projeto

MVP de compliance preditivo ferroviário desenvolvido em Python + Streamlit.

Dados 100% simulados para fins acadêmicos/demonstrativos.

Projeto de Allan Sabá — estudante de Engenharia Ferroviária e Logística na UFPA.



\## Stack

\- Python 3.11+

\- Streamlit (interface web)

\- SQLite via sqlite3 nativo (banco local em data/railguard.db)

\- Pandas + NumPy (dados)

\- Plotly (gráficos dark theme corporativo)

\- Scikit-learn RandomForestClassifier (modelo preditivo)



\## Estrutura de arquivos

\- app.py          → interface completa, roteamento, todas as páginas

\- database.py     → banco SQLite, CRUD, todas as queries

\- models.py       → constantes de domínio (pesos, listas, cores)

\- risk\_engine.py  → motor de risco: calcular\_risco\_operacional(), calcular\_rcrs()

\- ml\_model.py     → RandomForest: treino, predição, importância de features

\- reports.py      → geração de relatórios e exportação CSV

\- seed\_data.py    → dados fictícios de demonstração



\## Tema visual

Dark corporativo ferroviário. Paleta principal:

\- PRIMARY   = "#080F1D"  (fundo)

\- CARD\_BG   = "#101F38"  (cards)

\- ACCENT2   = "#2A8FD4"  (azul elétrico — info)

\- C\_CRITICO = "#E63946"  (vermelho — crítico)

\- C\_BAIXO   = "#0CB87A"  (verde — ok)

\- C\_MEDIO   = "#F4A62A"  (âmbar — atenção)

\- C\_ALTO    = "#F47B35"  (laranja — alto risco)

Todo CSS é injetado via st.markdown() com unsafe\_allow\_html=True.



\## Navegação

Feita via st.session\_state\["pagina"] + dicionário \_ROTAS no final do app.py.

Não usa st.sidebar.radio nativo — usa botões customizados com CSS.



\## Deploy

Streamlit Cloud em: https://github.com/sabazord/Railguard

Branch: main

Para subir alterações: git add . → git commit -m "msg" → git push origin main



\## Bugs já resolvidos (não reverter)

1\. Heatmap KeyError tipo\_ativo: df\_riscos já vem com tipo\_ativo do JOIN

&#x20;  no database.py — NÃO fazer merge com df\_ativos para esse campo.

&#x20;  Solução: if "tipo\_ativo" in df\_riscos.columns: usar diretamente.



2\. Score lookup nos alertas: usar df\_riscos\["ativo\_codigo"] == ativo\_cod

&#x20;  dentro de try/except, nunca df\_riscos.get().



\## Roadmap (o que ainda NÃO foi feito)

\- v0.4: Autenticação (streamlit-authenticator)

\- v0.5: Migração para PostgreSQL + Docker

\- v0.6: API REST com FastAPI para sensores IoT

\- v0.7: Explicabilidade via SHAP (estrutura já preparada em ml\_model.py)

\- v0.8: Mapa geoespacial dos trechos (folium/pydeck)

\- v1.0: Deploy em nuvem (AWS/Azure) com CI/CD



\## Próxima melhoria sugerida

Implementar SHAP em ml\_model.py — a estrutura já está nos comentários.

Adicionar nova função explicar\_com\_shap() e exibir na página Modelo Preditivo.



\## Como rodar localmente

streamlit run app.py

