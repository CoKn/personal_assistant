import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationSummaryMemory
import dotenv
import time


def chat():
    if "conversation" not in st.session_state:
        st.warning("No memory found. Please go to the embeddings page and load a document.")
    else:
        history = st.session_state.conversation.memory.chat_memory.messages

        for message in history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)

        if prompt := st.chat_input("What is up?"):

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.conversation.run(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.015)

                message_placeholder.markdown(full_response)


def embeddings():

    # url = st.text_input("Enter a url", key="url")
    # https://lilianweng.github.io/posts/2023-06-23-agent/

    if url := st.text_input("Enter a url", key="url"):
        with st.spinner("Loading..."):
            loader = WebBaseLoader(url)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            all_splits = text_splitter.split_documents(data)

            # vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=st.session_state.embedding_function
                )

            # retriever
            if "retriever" not in st.session_state:
                st.session_state.retriever = st.session_state.vectorstore.as_retriever()

            # conversation
            if "conversation" not in st.session_state:
                # st.session_state.conversation = ConversationChain(llm=st.session_state.gpt_35_turbo)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(st.session_state.gpt_35_turbo,
                                                                                      retriever=st.session_state.retriever,
                                                                                      memory=st.session_state.memory)

        st.success("Website loaded successfully")


if __name__ == '__main__':
    dotenv.load_dotenv()

    # llm
    if "gpt_35_turbo" not in st.session_state:
        st.session_state.gpt_35_turbo = ChatOpenAI(streaming=True,
                                                   model="gpt-3.5-turbo",
                                                   callbacks=[StreamingStdOutCallbackHandler()],
                                                   temperature=0)

    # embedding function
    if "embedding_function" not in st.session_state:
        st.session_state.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.gpt_35_turbo, memory_key="chat_history", return_messages=True
        )

    # pages
    pages = {
        "chat": chat,
        "embeddings": embeddings,
    }
    page_selection = st.sidebar.radio("Models", list(pages.keys()))
    selected_page = pages[page_selection]

    selected_page()

