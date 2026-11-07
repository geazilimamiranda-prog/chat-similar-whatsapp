import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# Configuração da página no padrão amplo
st.set_page_config(
    page_title="WhatsApp Assistant - Inversores",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS para visual idêntico ao WhatsApp Web
st.markdown("""
<style>
    /* Esconder elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Fundo da área de chat */
    .stApp {
        background-color: #0b141a;
    }
    
    /* Mensagem do Usuário (Verde WhatsApp) */
    .user-msg {
        background-color: #005c4b;
        color: #e9edef;
        padding: 10px 14px;
        border-radius: 8px 0px 8px 8px;
        margin: 5px 0 5px auto;
        max-width: 75%;
        width: fit-content;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14.5px;
        box-shadow: 0 1px 0.5px rgba(11,20,26,.13);
    }
    
    /* Mensagem do Assistente (Cinza WhatsApp) */
    .assistant-msg {
        background-color: #202c33;
        color: #e9edef;
        padding: 10px 14px;
        border-radius: 0px 8px 8px 8px;
        margin: 5px auto 5px 0;
        max-width: 75%;
        width: fit-content;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14.5px;
        box-shadow: 0 1px 0.5px rgba(11,20,26,.13);
    }
    
    /* Carimbo de hora */
    .timestamp {
        font-size: 10px;
        color: #8696a0;
        float: right;
        margin-left: 10px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização de Variáveis de Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# --- SIDEBAR: Configurações e Upload ---
with st.sidebar:
    st.title("💬 WhatsApp Config")
    st.subheader("1. Chave de API")
    
    # Recupera a chave das variáveis do Streamlit Secrets ou do input do usuário
    api_key = st.text_input("OpenAI API Key", type="password", help="Insira sua chave sk-...")
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    st.divider()
    st.subheader("2. Upload de Arquivo")
    uploaded_file = st.file_uploader(
        "Carregue a planilha ou guia (XLSX, CSV, TXT)", 
        type=["xlsx", "csv", "txt"]
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                st.session_state.file_context = df.to_string()
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.session_state.file_context = df.to_string()
            else:
                st.session_state.file_context = uploaded_file.read().decode("utf-8")
                
            st.session_state.file_name = uploaded_file.name
            st.success(f"✅ Arquivo ativo: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            
    if st.session_state.file_name:
        st.info(f"📄 **Base ativa:** {st.session_state.file_name}")
        
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

# --- HEADER DO CHAT (Estilo WhatsApp) ---
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("### 🟢")
with col2:
    st.markdown("**Especialista em Inversores de Frequência**\n\n*Online - Resposta baseada no arquivo*")

st.divider()

# --- EXIBIÇÃO HISTÓRICO DE MENSAGENS ---
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{message["content"]}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-msg">{message["content"]}</div>', 
                unsafe_allow_html=True
            )

# --- CAMPO DE ENTRADA E PROCESSAMENTO ---
if prompt := st.chat_input("Digite uma mensagem..."):
    if not api_key:
        st.warning("Por favor, insira a sua API Key da OpenAI na barra lateral.")
        st.stop()
        
    # Registrar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)
    
    # Preparar contexto do sistema
    system_prompt = (
        "Você é um consultor especialista em manutenção industrial e inversores de frequência. "
        "Responda às dúvidas do usuário com base estrita no contexto do arquivo fornecido abaixo. "
        "Responda com objetividade, clareza e no idioma Português.\n\n"
        f"--- CONTEXTO DO ARQUIVO ({st.session_state.file_name or 'Sem arquivo'}) ---\n"
        f"{st.session_state.file_context[:150000]}" # Limite preventivo de caracteres
    )
    
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Chamada à API
    client = OpenAI(api_key=api_key)
    
    with st.spinner("digitando..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                temperature=0.2
            )
            bot_reply = response.choices[0].message.content
            
            # Salvar e exibir resposta do robô
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.markdown(f'<div class="assistant-msg">{bot_reply}</div>', unsafe_allow_html=True)
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao comunicar com a API: {e}")
