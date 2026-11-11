import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuração da página em modo wide
st.set_page_config(page_title="Painel de Oficina Mecânica & Assistente IA", layout="wide", page_icon="🔧")

# ==========================================
# 1. INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "df" not in st.session_state:
    st.session_state.df = None

# ==========================================
# 2. ESTILIZAÇÃO CSS CUSTOMIZADA
# ==========================================
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #a82020;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #8a1919;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BARRA LATERAL (CONFIGURAÇÕES DA AZURE)
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Credenciais Azure OpenAI:")
    azure_api_key = st.text_input("Azure OpenAI API Key", type="password", value=os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", "https://"), placeholder="https://seu-recurso.openai.azure.com/")
    
    st.markdown("### Deployment / Modelo:")
    azure_deployment = st.text_input("Deployment", value="gpt-4.1-mini")
    
    azure_api_version = "2024-02-15-preview"
    embedding_deployment = "text-embedding-ada-002"

    st.markdown("---")
    st.markdown("### 🎭 Persona da IA (Oficina):")
    default_persona = (
        "Você é um consultor técnico sênior de uma oficina mecânica automotiva. "
        "Responda de forma ágil, clara e cordial. Consulte sempre a base de dados de ordens de serviço, "
        "peças e manutenções para fornecer orçamentos precisos, diagnósticos e prazos."
    )
    persona = st.text_area("Persona", value=default_persona, height=120)

    st.markdown("---")
    st.markdown("### 📁 Upload da Base (CSV da Oficina):")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"Carregado: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão de Conexão com validação
    if st.button("🚀 Conectar IA", use_container_width=True):
        if not azure_api_key or not azure_endpoint:
            st.error("Preencha a API Key e o Endpoint da Azure!")
        else:
            st.session_state.chat_active = True
            st.success("IA da Oficina Conectada com Sucesso!")

    # Botão de Novo Chat alterado de Checkbox para Botão Real
    if st.button("🔄 Novo Chat", use_container_width=True):
        st.session_state.messages = []
        st.success("Novo chat iniciado!")
        st.rerun()

# ==========================================
# 4. PROCESSAMENTO DO RAG (BASE DA OFICINA)
# ==========================================
def process_dataframe_to_vectorstore(df, api_key, endpoint, api_version, emb_deployment):
    try:
        docs = []
        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=row_text))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents
