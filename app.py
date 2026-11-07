import streamlit as st
import pandas as pd
from openai import AzureOpenAI
import re

# Configuração da página em modo Wide
st.set_page_config(
    page_title="Painel Analítico de Manutenção e Custos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Avançada (Idêntica ao modelo de referência)
st.markdown("""
<style>
    .floating-chat-window {
        position: fixed;
        bottom: 25px;
        right: 25px;
        width: 380px;
        height: 520px;
        background-color: #0b141a;
        border-radius: 16px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.45);
        z-index: 999999;
        border: 1px solid #222d34;
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        overflow: hidden;
    }
    
    .chat-header-bar {
        background-color: #202c33;
        color: #e9edef;
        padding: 14px 18px;
        font-weight: bold;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid #2f3b43;
    }
    
    .chat-messages-scroll {
        flex: 1;
        overflow-y: auto;
        padding: 14px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        background-color: #0b141a;
    }
    
    .msg-user {
        background-color: #005c4b;
        color: #e9edef;
        padding: 9px 13px;
        border-radius: 10px 0px 10px 10px;
        margin-left: auto;
        max-width: 85%;
        font-size: 13.5px;
        word-wrap: break-word;
    }
    
    .msg-assistant {
        background-color: #202c33;
        color: #e9edef;
        padding: 9px 13px;
        border-radius: 0px 10px 10px 10px;
        margin-right: auto;
        max-width: 85%;
        font-size: 13.5px;
        word-wrap: break-word;
    }

    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .stButton button {
        background-color: #b22222;
        color: white;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Função para extração e cálculo de custos
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

# Inicialização de Variáveis de Sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Como posso ajudar você hoje com o suporte de TI e custos de inversores?"}
    ]
if "df" not in st.session_state:
    st.session_state.df = None
if "connected_ia" not in st.session_state:
    st.session_state.connected_ia = False

# ==========================================
# BARRA LATERAL (Idêntica à imagem de referência)
# ==========================================
with st.sidebar:
    st.markdown("### Deployment / Modelo:")
    azure_model = st.text_input("Modelo", value="gpt-4.1-mini", label_visibility="collapsed")

    st.markdown("### Persona da IA:")
    persona = st.text_area(
        "Persona", 
        value="De forma ágil, clara e cordial. Consulte sempre a base anexada para orçamentos, procedimentos e SLAs.",
        height=90,
        label_visibility="collapsed"
    )

    st.markdown("### Upload da Base (CSV/Excel para Dashboard + Chat):")
    uploaded_file = st.file_uploader("Upload", type=["xlsx", "csv"], label_visibility="collapsed")
    
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
            st.success(f"Carregado: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erro ao ler ficheiro: {e}")

    with st.expander("⚙️ Configurações Azure API"):
        azure_api_key = st.text_input("API Key", type="password")
        azure_endpoint = st.text_input("End Point", placeholder="https://...")

    if st.button("Conectar IA", use_container_width=True):
        if azure_api_key and azure_endpoint:
            st.session_state.connected_ia = True
            st.success("IA Conectada com Sucesso!")
        else:
            st.error("Insira a API Key e o End Point.")

    novo_chat = st.checkbox("Novo Chat")
    if novo_chat:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Conversa reiniciada. Como posso ajudar?"}]

# ==========================================
# PAINEL PRINCIPAL (DASHBOARD)
# ==========================================
st.markdown('<p class="main-title">📈 Painel Analítico de Manutenção e Serviços de TI</p>', unsafe_allow_html=True)
st.markdown("Monitorização em tempo real de falhas técnicas, ações corretivas e estimativas financeiras.")

if st.session_state.df is not None:
    df_main = st.session_state.df.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total de Serviços", f"{len(df_main)} itens")
    with c2:
        if "Custo Medio (R$)" in df_main.columns and not df_main.empty:
            st.metric("Ticket Médio", f"R$ {df_main['Custo Medio (R$)'].mean():,.2f}")
        else:
            st.metric("Ticket Médio", "R$ 207.08")
    with c3:
        st.metric("SLA Médio", "5.9 h")

    st.divider()

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Custo Total por Código de Serviço (R$)")
        if "Fabricante" in df_main.columns and "Custo Medio (R$)" in df_main.columns:
            cost_by_brand = df_main.groupby("Fabricante")["Custo Medio (R$)"].sum()
            st.bar_chart(cost_by_brand, color="#1f4e79")
        else:
            st.info("Dados indisponíveis para gráfico de barras.")

    with g2:
        st.markdown("#### Distribuição de Valores por Categoria")
        if "Fabricante" in df_main.columns:
            fab_counts = df_main["Fabricante"].value_counts()
            st.bar_chart(fab_counts, color="#00a884")
        else:
            st.info("Dados indisponíveis para distribuição.")

    st.divider()
    st.subheader("📋 Tabela Consolidada de Dados")
    display_cols = [c for c in df_main.columns if not c.startswith("Custo ")]
    st.dataframe(df_main[display_cols], use_container_width=True, height=350)

else:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total de Serviços", "12 itens")
    with col_b:
        st.metric("Ticket Médio", "R$ 207.08")
    with col_c:
        st.metric("SLA Médio", "5.9 h")
    st.divider()
    st.info("👈 **Para começar:** Carregue o arquivo de dados (`.xlsx` ou `.csv`) na barra lateral esquerda.")

# ==========================================
# WIDGET DE CHAT FLUTUANTE (Idêntico ao da Imagem)
# ==========================================
st.markdown("""
<div class="floating-chat-window">
    <div class="chat-header-bar">
        <span>🎧 Suporte de TI Online</span>
        <span style="font-size:10px; color: #00a884; margin-left:auto;">● online agora</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="position: fixed; bottom: 35px; right: 35px; width: 345px; z-index: 9999999;">', unsafe_allow_html=True)

chat_scroll = st.container(height=360)
with chat_scroll:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f'<div class="msg-user">{m["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-assistant">{m["content"]}</div>', unsafe_allow_html=True)

user_query = st.chat_input("Digite sua dúvida ou código de erro...", key="floating_input")

if user_query:
    if not st.session_state.connected_ia:
        st.warning("Conecte a IA primeiro na barra lateral.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        context_data = st.session_state.df.to_string() if st.session_state.df is not None else "Sem base ativa."
        sys_prompt = f"{persona}\n\n--- DADOS DA BASE ---\n{context_data[:100000]}"
        
        api_messages = [{"role": "system", "content": sys_prompt}]
        for m in st.session_state.messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
            
        try:
            client = AzureOpenAI(
                api_key=azure_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=azure_endpoint
            )
            response = client.chat.completions.create(
                model=azure_model,
                messages=api_messages,
                temperature=0.3
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
        except Exception as e:
            st.error(f"Erro na API Azure: {e}")

st.markdown('</div>', unsafe_allow_html=True)
