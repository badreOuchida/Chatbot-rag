import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader


st.set_page_config(page_title="Chat with CV docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
openai.api_key = st.secrets["openai_key"]["key"]
col1, col2, col3 = st.columns([1, 1.5, 1])
with col1:
    st.image("Logo_inpt.PNG", width=150)
with col3:
    st.image("llamaindex_logo.png", width=180)
st.title("Chatbot utilisant RAG et LLM pour le code de travail Marocain en Python avec LlamaIndex et Streamlit.")
st.info("Projet réalisé par Taqi Anas et Ouchida Badreddine", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Posez-moi des questions sur le code de travail Marocain!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Chargement  ... (Un premier chargement peut prendre du temps pour l'indexations)"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert o$
        # index = VectorStoreIndex.from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Vous êtes un expert dans le code de travail au Maroc et votre travail consiste à répondre aux questions. Gardez vos réponses basées sur des faits - ne hallucinez pas de fonctionnalités"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Votre question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Chargement..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
