import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Configuração da página em modo wide
st.set_page_config(page_title="Oficina Mecânica - Gestão & IA", layout="wide", page_icon="🔧")

# ==========================================
# 1. INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# ==========================================
# 2. ESTILIZAÇÃO CSS (JANELA FLUTUANTE DO WHATSAPP)
# ==========================================
st.markdown("""
<style>
    /* Estilização dos Botões da Barra Lateral */
    div.stButton > button:first-child {
        background-color: #075E54;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #128C7E;
        color: white;
    }

    /* Posicionamento Fixo da Janela Flutuante no Canto Inferior Direito */
    .wa-floating-container {
        position: fixed;
        bottom: 20px;
        right: 30px;
        width: 380px;
        background-color: #efeae2;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        z-index: 999999;
        overflow: hidden;
        border: 1px solid #d1c7bd;
    }
    
    .wa-header {
        background-color: #075E54;
        color: white;
        padding: 12px 16px;
        font-weight: bold;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .wa-body {
        padding: 10px;
        height: 260px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        background-color: #efeae2;
    }

    .chat-bubble-user {
        background-color: #dcf8c6;
        color: #000;
        padding: 8px 12px;
        border-radius: 8px 0px 8px 8px;
        margin: 4px 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 1px 1px rgba(0,0,0,0.15);
        font-size: 13px;
        word-wrap: break-word;
    }

    .chat-bubble-ai {
        background-color: #ffffff;
        color: #000;
        padding: 8px 12px;
        border-radius: 0px 8px 8px 8px;
        margin: 4px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 1px 1px rgba(0,0,0,0.15);
        font-size: 13px;
        word-wrap: break-word;
    }
    
    .wa-footer {
        background-color: #f0f0f0;
        padding: 8px 12px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BARRA LATERAL (CREDENCIAIS E UPLOAD)
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Credenciais Azure OpenAI:")
    azure_api_key = st.text_input("Azure OpenAI API Key", type="password", value=os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", "https://"), placeholder="https://seu-recurso.openai.azure.com/")
    
    azure_deployment = st.text_input("Deployment", value="gpt-4.1-mini")
    azure_api_version = "2024-02-15-preview"

    st.markdown("---")
    st.markdown("### 🎭 Persona da IA (Oficina):")
    default_persona = (
        "Você é um consultor técnico sênior de uma oficina mecânica automotiva. "
        "Responda de forma ágil, clara e cordial estilo WhatsApp. Consulte sempre a base de dados de ordens de serviço, "
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
    
    if st.button("🚀 Conectar IA", use_container_width=True):
        if not azure_api_key or not azure_endpoint:
            st.error("Preencha a API Key e o Endpoint da Azure!")
        else:
            st.success("IA Conectada com Sucesso!")

    if st.button("🔄 Novo Chat", use_container_width=True):
        st.session_state.messages = []
        st.success("Novo chat iniciado!")
        st.rerun()

# ==========================================
# 4. FUNÇÃO PARA BUSCAR CONTEXTO NO CSV
# ==========================================
def buscar_contexto_csv(df, query):
    if df is None or df.empty:
        return "Nenhuma base de dados carregada."
    
    query_terms = query.lower().split()
    matched_rows = []
    
    for _, row in df.iterrows():
        row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
        if any(term in row_str.lower() for term in query_terms):
            matched_rows.append(row_str)
            
    if matched_rows:
        return "\n".join(matched_rows[:5])
    else:
        return df.to_string(index=False)

# ==========================================
# 5. PAINEL ANALÍTICO PRINCIPAL (DASHBOARD)
# ==========================================
st.markdown("## 🚗 Painel de Controle e Gestão da Oficina Mecânica")

if st.session_state.df is not None:
    df = st.session_state.df
    
    col_servico = "Tipo de Serviço" if "Tipo de Serviço" in df.columns else df.columns[0]
    col_valor = "Valor do Serviço (R$)" if "Valor do Serviço (R$)" in df.columns else (df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else None)

    total_servicos = len(df)
    ticket_medio = df[col_valor].mean() if col_valor and col_valor in df.columns else 0.0
    soma_total = df[col_valor].sum() if col_valor and col_valor in df.columns else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Serviços Cadastrados", f"{total_servicos} itens")
    with col2:
        st.metric("Média de Preço dos Serviços", f"R$ {ticket_medio:,.2f}")
    with col3:
        st.metric("Valor Total do Portfólio", f"R$ {soma_total:,.2f}")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🛠️ Valor por Tipo de Serviço (R$)")
        if col_valor and col_servico in df.columns:
            chart_data = df.set_index(col_servico)[[col_valor]]
            st.bar_chart(chart_data)
        else:
            st.bar_chart(df.select_dtypes(include=['number']))
            
    with c2:
        st.markdown("#### 🥧 Distribuição de Custos (Gráfico de Pizza)")
        if col_valor and col_servico in df.columns:
            fig, ax = plt.subplots(figsize=(5, 5))
            dados_pizza = df.nlargest(8, col_valor) if len(df) > 8 else df
            labels = dados_pizza[col_servico].values
            valores = dados_pizza[col_valor].values
            colors = plt.cm.tab10.colors[:len(labels)]
            
            ax.pie(
                valores, 
                labels=labels, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors,
                textprops={'fontsize': 8}
            )
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("Dados insuficientes para gerar o gráfico de pizza.")

    with st.expander("🔍 Visualizar Tabela Completa de Serviços e Peças"):
        st.dataframe(df, use_container_width=True)

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Serviços Cadastrados", "12 itens")
    with col2:
        st.metric("Média de Preço dos Serviços", f"R$ 207.08")
    with col3:
        st.metric("Valor Total do Portfólio", f"R$ 2,485.00")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🛠️ Valor por Tipo de Serviço (R$)")
        sample_data = pd.DataFrame({
            "Troca de Óleo": 80, 
            "Pastilhas de Freio": 120, 
            "Alinhamento 3D": 150, 
            "Correia Dentada": 350, 
            "Embreagem": 450
        }, index=["Valor (R$)"]).T
        st.bar_chart(sample_data)
    with c2:
        st.markdown("#### 🥧 Distribuição de Custos (Gráfico de Pizza)")
        fig, ax = plt.subplots(figsize=(5, 5))
        sample_values = [80, 120, 150, 350, 450]
        sample_labels = ["Troca de Óleo", "Pastilhas", "Alinhamento", "Correia", "Embreagem"]
        ax.pie(sample_values, labels=sample_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        ax.axis('equal')
        st.pyplot(fig)

# ==========================================
# 6. WIDGET FLUTUANTE 100% INTEGRADO (WHATSAPP)
# ==========================================

# Renderiza a estrutura HTML fixa do card no canto inferior direito
st.markdown("""
<div class="wa-floating-container">
    <div class="wa-header">
        <span>🟢</span> <span>Oficina Mecânica - Suporte WhatsApp</span>
    </div>
    <div class="wa-body">
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown('<div class="chat-bubble-ai">🤖 Olá! Sou o assistente virtual da oficina via WhatsApp. Como posso ajudar?</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f'<div class="chat-bubble-user">{msg.content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-ai">{msg.content}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Rodapé do card contendo o formulário de envio perfeitamente encaixado
st.markdown('<div class="wa-footer">', unsafe_allow_html=True)

with st.form(key="wa_widget_form", clear_on_submit=True):
    col_input, col_send = st.columns([4, 1])
    with col_input:
        user_input = st.text_input("Mensagem", placeholder="Digite...", label_visibility="collapsed")
    with col_send:
        submit_btn = st.form_submit_button("Enviar")

st.markdown('</div></div>', unsafe_allow_html=True)

# Processamento da mensagem enviada pelo formulário integrado
if submit_btn and user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    if not azure_api_key or not azure_endpoint:
        answer = "⚠️ Por favor, configure sua API Key e Endpoint da Azure na barra lateral para falar comigo."
    else:
        try:
            llm = AzureChatOpenAI(
                azure_deployment=azure_deployment.strip(),
                azure_endpoint=azure_endpoint.strip(),
                api_key=azure_api_key.strip(),
                api_version=azure_api_version,
                temperature=0.2
            )
            
            contexto_csv = buscar_contexto_csv(st.session_state.df, user_input)
            
            system_prompt = (
                f"{persona}\n\n"
                f"Utilize os dados da tabela da oficina abaixo para responder de forma direta e amigável:\n"
                f"--- CONTEXTO DO CSV ---\n{contexto_csv}\n------------------------"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            
            answer = chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages
            })
            
        except Exception as e:
            answer = f"❌ Erro de conexão com a Azure OpenAI: `{e}`"

    st.session_state.messages.append(AIMessage(content=answer))
    st.rerun()
