import streamlit as st
import pandas as pd
from openai import AzureOpenAI
import re

# Configuração da página
st.set_page_config(
    page_title="Gestão de Falhas & Custos - Inversores",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada (Estilo WhatsApp + Cards Industriais)
st.markdown("""
<style>
    /* Fundo da Barra Lateral */
    [data-testid="stSidebar"] {
        background-color: #111b21;
    }
    
    /* Balões do Chat (Estilo WhatsApp) */
    .user-msg {
        background-color: #005c4b;
        color: #e9edef;
        padding: 10px 14px;
        border-radius: 8px 0px 8px 8px;
        margin: 6px 0 6px auto;
        max-width: 88%;
        font-size: 13.5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .assistant-msg {
        background-color: #202c33;
        color: #e9edef;
        padding: 10px 14px;
        border-radius: 0px 8px 8px 8px;
        margin: 6px auto 6px 0;
        max-width: 88%;
        font-size: 13.5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .chat-header {
        color: #00a884;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Função para tratar custos
def parse_costs(val):
    if pd.isna(val):
        return 0.0, 0.0, 0.0
    val_str = str(val).replace("R$", "").replace(".", "").replace(",", ".").strip()
    numbers = re.findall(r'\d+\.?\d*', val_str)
    if len(numbers) >= 2:
        c_min, c_max = float(numbers[0]), float(numbers[1])
    elif len(numbers) == 1:
        c_min = c_max = float(numbers[0])
    else:
        c_min = c_max = 0.0
    return c_min, c_max, (c_min + c_max) / 2.0

# Inicialização da Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# ==========================================
# BARRA LATERAL: CONFIGURAÇÕES AZURE E CHAT
# ==========================================
with st.sidebar:
    st.markdown('<div class="chat-header">💬 WhatsApp Assistente Técnico</div>', unsafe_allow_html=True)
    st.caption("Especialista em manutenção e diagnóstico (Azure OpenAI).")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Credenciais Azure OpenAI")
    azure_api_key = st.text_input("API Key", type="password", help="Chave de API da Azure OpenAI")
    azure_endpoint = st.text_input("End Point", placeholder="https://seu-recurso.openai.azure.com/", help="URL do Endpoint da Azure")
    azure_model = st.text_input("Modelo (Deployment Name)", value="gpt-4o-mini", help="Nome do deployment configurado no Azure")

    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Carregar Base (.xlsx, .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                try:
                    df_temp = pd.read_excel(uploaded_file, header=3)
                    if "Fabricante" not in df_temp.columns:
                        df_temp = pd.read_excel(uploaded_file)
                except:
                    df_temp = pd.read_excel(uploaded_file)
            else:
                df_temp = pd.read_csv(uploaded_file)

            if "Custo Total Estimado (R$)" in df_temp.columns:
                parsed = df_temp["Custo Total Estimado (R$)"].apply(parse_costs)
                df_temp["Custo Min (R$)"] = [p[0] for p in parsed]
                df_temp["Custo Max (R$)"] = [p[1] for p in parsed]
                df_temp["Custo Medio (R$)"] = [p[2] for p in parsed]

            st.session_state.df = df_temp
            st.success("Base carregada com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar ficheiro: {e}")

    st.divider()
    st.markdown("**Conversa WhatsApp**")
    chat_box = st.container(height=320)
    
    with chat_box:
        for m in st.session_state.messages:
            role_class = "user-msg" if m["role"] == "user" else "assistant-msg"
            st.markdown(f'<div class="{role_class}">{m["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Pergunte sobre um erro ou custo..."):
        if not azure_api_key or not azure_endpoint:
            st.warning("Preencha a API Key e o End Point da Azure na barra lateral.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            context_str = st.session_state.df.to_string() if st.session_state.df is not None else "Nenhum ficheiro ativo."
            system_prompt = (
                "Você é
