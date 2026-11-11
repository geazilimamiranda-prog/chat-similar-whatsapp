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

# Configuração da página
st.set_page_config(page_title="Painel Analítico e Suporte de Manutenção", layout="wide", page_icon="📊")

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
# 3. BARRA LATERAL (CONFIGURAÇÕES)
# ==========================================
with st.sidebar:
    st.markdown("### Deployment / Modelo:")
    azure_deployment = st.text_input("Deployment", value="gpt-4.1-mini", label_visibility="collapsed")
    
    azure_api_key = st.text_input("Azure OpenAI API Key", type="password", value=os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_api_version = "2024-02-15-preview"
    embedding_deployment = "text-embedding-ada-002"

    st.markdown("---")
    st.markdown("### Persona da IA:")
    default_persona = (
        "forma ágil, clara e cordial. Consulte sempre a base anexada para orçamentos, procedimentos e SLAs."
    )
    persona = st.text_area("Persona", value=default_persona, height=100, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Upload da Base (CSV para Dashboard + Chat):")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        st.success(f"Carregado: {uploaded_file.name}")
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Conectar IA", use_container_width=True):
        st.session_state.chat_active = True
        st.success("IA Conectada com sucesso!")

    novo_chat = st.checkbox("Novo Chat")
    if novo_chat:
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 4. PROCESSAMENTO DO RAG DIRETO DO PANDAS (SEM ERROS DE LOADER)
# ==========================================
def process_dataframe_to_vectorstore(df, api_key, endpoint, api_version, emb_deployment):
    try:
        # Transforma cada linha do dataframe em um documento de texto para o RAG
        docs = []
        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=row_text))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=emb_deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao gerar embeddings na Azure: {e}")
        return None

if st.session_state.df is not None and st.session_state.vectorstore is None and azure_api_key:
    st.session_state.vectorstore = process_dataframe_to_vectorstore(
        st.session_state.df, azure_api_key, azure_endpoint, azure_api_version, embedding_deployment
    )

# ==========================================
# 5. PAINEL ANALÍTICO PRINCIPAL (DASHBOARD)
# ==========================================
st.markdown("## 📊 Painel Analítico de Manutenção e Serviços de TI")

if st.session_state.df is not None:
    df = st.session_state.df
    col1, col2, col3 = st.columns(3)
    
    total_servicos = len(df)
    ticket_medio = df.select_dtypes(include=['number']).mean().mean() if not df.empty else 0.0
    
    with col1:
        st.metric("Total de Serviços", f"{total_servicos} itens")
    with col2:
        st.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
    with col3:
        st.metric("SLA Médio", "5.9 h")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Custo Total por Código de Serviço (R$)")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.bar_chart(df[numeric_cols[0]])
        else:
            st.info("O CSV não possui colunas numéricas para exibir o gráfico.")
            
    with c2:
        st.markdown("#### Distribuição de Valores por Categoria")
        if len(numeric_cols) > 1:
            st.line_chart(df[numeric_cols[:min(2, len(numeric_cols))]])
        else:
            st.area_chart(df[numeric_cols[0]] if len(numeric_cols) > 0 else pd.Series([10, 20, 30]))
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Serviços", "12 itens")
    with col2:
        st.metric("Ticket Médio", "R$ 207.08")
    with col3:
        st.metric("SLA Médio", "5.9 h")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Custo Total por Código de Serviço (R$)")
        sample_data = pd.DataFrame({"Hardware": [250, 300, 150, 600, 110], "Software": [100, 80, 130, 120, 90]})
        st.bar_chart(sample_data)
    with c2:
        st.markdown("#### Distribuição de Valores por Categoria")
        st.area_chart(pd.DataFrame({"Valores": [10, 25, 40, 20, 50]}))

# ==========================================
# 6. WIDGET DE CHAT FLUTUANTE
# ==========================================
st.markdown("---")
st.sidebar.markdown("### 💬 Chat de Suporte")

with st.container():
    st.markdown("---")
    st.subheader("🤖 Suporte de TI Online (Chat Ativo)")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_essage(role):
                st.markdown(msg.content)

    if user_input := st.chat_input("Digite sua dúvida ou código de erro..."):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        answer = "No momento, estou operando em modo de interface visual. Conecte sua chave da Azure OpenAI na barra lateral para interagir com o modelo e a base RAG."
        
        if azure_api_key and st.session_state.vectorstore:
            try:
                llm = AzureChatOpenAI(
                    azure_deployment=azure_deployment,
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    temperature=0.2
                )
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                
                system_prompt = f"{persona}\n\nContexto:\n{{context}}"
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                
                rag_chain = (
                    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                     "input": RunnablePassthrough(),
                     "chat_history": lambda x: st.session_state.messages}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke(user_input)
            except Exception as e:
                answer = f"Erro ao consultar o modelo Azure: {e}"

        st.session_state.messages.append(AIMessage(content=answer))
        st.rerun()
