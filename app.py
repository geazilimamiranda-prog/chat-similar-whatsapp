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

    # Botão de Novo Chat
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
        splits = text_splitter.split_documents(docs)

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=emb_deployment,
            azure_endpoint=endpoint.strip(),
            api_key=api_key.strip(),
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
st.markdown("## 🚗 Painel de Controle e Gestão da Oficina Mecânica")

if st.session_state.df is not None:
    df = st.session_state.df
    col1, col2, col3 = st.columns(3)
    
    total_ordens = len(df)
    ticket_medio = df.select_dtypes(include=['number']).mean().mean() if not df.empty else 0.0
    
    with col1:
        st.metric("Ordens de Serviço", f"{total_ordens} O.S.")
    with col2:
        st.metric("Ticket Médio (Reparos)", f"R$ {ticket_medio:,.2f}")
    with col3:
        st.metric("Tempo Médio de Box", "4.5 h")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Custo de Peças e Serviços por Código (R$)")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.bar_chart(df[numeric_cols[0]])
        else:
            st.info("O CSV não possui colunas numéricas para exibir o gráfico.")
            
    with c2:
        st.markdown("#### Fluxo de Entrada por Categoria")
        if len(numeric_cols) > 1:
            st.line_chart(df[numeric_cols[:min(2, len(numeric_cols))]])
        else:
            st.area_chart(df[numeric_cols[0]] if len(numeric_cols) > 0 else pd.Series([100, 250, 400]))
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ordens de Serviço", "28 O.S.")
    with col2:
        st.metric("Ticket Médio (Reparos)", "R$ 485.50")
    with col3:
        st.metric("Tempo Médio de Box", "4.2 h")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Custo de Peças e Serviços por Código (R$)")
        sample_data = pd.DataFrame({
            "Freios e Suspensão": [350, 600, 220, 800, 410], 
            "Motor e Injeção": [900, 450, 1200, 750, 620]
        })
        st.bar_chart(sample_data)
    with c2:
        st.markdown("#### Fluxo de Entrada por Categoria")
        st.area_chart(pd.DataFrame({"Atendimentos": [5, 12, 8, 20, 15]}))

# ==========================================
# 6. WIDGET DE CHAT FLUTUANTE / SUPORTE MECÂNICO
# ==========================================
st.markdown("---")
st.sidebar.markdown("### 💬 Assistente Mecânico IA")

with st.container():
    st.markdown("---")
    st.subheader("🛠️ Atendimento e Diagnóstico Automotivo Online")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    if user_input := st.chat_input("Digite o sintoma do veículo, código de falha (OBD) ou peça..."):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        if not azure_api_key or not azure_endpoint:
            answer = "⚠️ **Atenção:** Por favor, insira a sua **API Key** e o **Endpoint da Azure** na barra lateral e clique em **Conectar IA**."
        else:
            try:
                llm = AzureChatOpenAI(
                    azure_deployment=azure_deployment.strip(),
                    azure_endpoint=azure_endpoint.strip(),
                    api_key=azure_api_key.strip(),
                    api_version=azure_api_version,
                    temperature=0.2
                )
                
                if st.session_state.vectorstore:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                    system_prompt = f"{persona}\n\nContexto da Oficina:\n{{context}}"
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
                else:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", persona),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])
                    chain = prompt | llm
                    response = chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.messages
                    })
                    answer = response.content
            except Exception as e:
                answer = f"❌ **Erro de comunicação com a Azure OpenAI:** `{e}`"

        st.session_state.messages.append(AIMessage(content=answer))
        st.rerun()
