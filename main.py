import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import dotenv
import time


def chat():
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


if __name__ == '__main__':
    dotenv.load_dotenv()

    # llm
    if "gpt_35_turbo" not in st.session_state:
        st.session_state.gpt_35_turbo = ChatOpenAI(streaming=True,
                                                   model="gpt-3.5-turbo",
                                                   callbacks=[StreamingStdOutCallbackHandler()],
                                                   temperature=0)
    if "lama" not in st.session_state:
        st.session_state.lama = None

    # conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(llm=st.session_state.gpt_35_turbo)

    chat()
