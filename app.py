import streamlit as st
import pandas as pd
from openai import OpenAI
import io

# Configuração da página em modo Wide (Amplo)
st.set_page_config(
    page_title="Dashboard & Chat - Inversores",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS para o Chat Estilo WhatsApp na Lateral e visual moderno
st.markdown("""
<style>
    /* Estilização da Barra Lateral (Chat) */
    [data-testid="stSidebar"] {
        background-color: #111b21;
        padding: 1rem;
    }
    
    /* Mensagens do usuário na barra lateral (Verde WhatsApp) */
    .user-msg-side {
        background-color: #005c4b;
        color: #e9edef;
        padding: 8px 12px;
        border-radius: 8px 0px 8px 8px;
        margin: 6px 0 6px auto;
        max-width: 85%;
        font-size: 13.5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Mensagens do assistente na barra lateral (Cinza WhatsApp) */
    .assistant-msg-side {
        background-color: #202c33;
        color: #e9edef;
        padding: 8px 12px;
        border-radius: 0px 8px 8px 8px;
        margin: 6px auto 6px 0;
        max-width: 85%;
        font-size: 13.5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Títulos do dashboard principal */
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização de Variáveis de Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_df" not in st.session_state:
    st.session_state.file_df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ==========================================
# BARRA LATERAL: CHAT ESTILO WHATSAPP
# ==========================================
with st.sidebar:
    st.markdown("### 💬 Assistente WhatsApp")
    st.markdown("<p style='font-size:12px; color: #8696a0;'>Especialista em Inversores & Custos</p>", unsafe_allow_html=True)
    
    # Campo para chave de API
    api_key = st.text_input("OpenAI API Key", type="password", help="Insira sua chave sk-...")
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    st.divider()
    
    # Upload de Arquivo
    st.markdown("#### 📂 Carregar Base de Dados")
    uploaded_file = st.file_uploader("Envie XLSX, CSV ou TXT", type=["xlsx", "csv", "txt"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                # Lê a planilha do excel gerada anteriormente
                st.session_state.file_df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                st.session_state.file_df = pd.read_csv(uploaded_file)
            else:
                content = uploaded_file.read().decode("utf-8")
                st.session_state.file_df = pd.DataFrame({"Conteúdo": [content]})
                
            st.session_state.file_name = uploaded_file.name
            st.success(f"Carregado: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            
    st.divider()
    
    # Caixa de Histórico de Conversas (Rolável na lateral)
    st.markdown("#### Conversa")
    chat_container = st.container(height=350)
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg-side">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-msg-side">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input de Mensagem do Chat na Lateral
    if prompt := st.chat_input("Pergunte algo sobre os dados..."):
        if not api_key:
            st.warning("Insira a API Key acima.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Contexto do arquivo atual
            df_text = st.session_state.file_df.to_string() if st.session_state.file_df is not None else "Nenhum arquivo carregado."
            system_prompt = (
                "Você é um consultor especialista em manutenção industrial e inversores de frequência. "
                "Responda com base nos dados da planilha fornecida.\n\n"
                f"--- DADOS ATUAIS ---\n{df_text[:100000]}"
            )
            
            api_messages = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.messages:
                api_messages.append({"role": m["role"], "content": m["content"]})
                
            client = OpenAI(api_key=api_key)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages,
                    temperature=0.2
                )
                bot_reply = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.rerun()
            except Exception as e:
                st.error(f"Erro na API: {e}")

    if st.button("Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# TELA PRINCIPAL: DASHBOARD & GRÁFICOS
# ==========================================
st.markdown('<p class="main-header">📊 Dashboard Analítico de Inversores e Custos de Manutenção</p>', unsafe_allow_html=True)
st.markdown("Visualize abaixo as métricas, gráficos de incidência e custos extraídos da sua base de dados industrial.")

if st.session_state.file_df is not None:
    df = st.session_state.file_df
    
    # Verifica se é a planilha padrão de guias e custos (contém colunas esperadas)
    expected_cols = ["Fabricante", "Custo Total Estimado (R$)"]
    has_cost_data = all(col in df.columns for col in expected_cols)
    
    # Métricas Principais no Topo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Falhas Mapeadas", len(df))
    with col2:
        if "Fabricante" in df.columns:
            st.metric("Fabricantes Atendidos", df["Fabricante"].nunique())
        else:
            st.metric("Linhas no Arquivo", len(df))
    with col3:
        if has_cost_data:
            # Limpa e calcula a média dos custos estimados se possível
            try:
                costs = df["Custo Total Estimado (R$)"].astype(str).str.split("-").str[0].str.replace(",", "."
