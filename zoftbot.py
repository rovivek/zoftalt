import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import pinecone
import os
import pymongo
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.chat_models.base import BaseChatModel

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

class ChatAnthropic(BaseChatModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # This import statement was moved to the bottom of the module
        from langchain.llms.base import LLM

        self.llm = LLM(**kwargs)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_pinconevectorstore(text_chunks):
    pinecone.init(
        api_key=st.secrets["zoft_pincone_API_KEY"],  # find at app.pinecone.io
        # next to api key in console
        environment=st.secrets["zoft_pincone_env"],
    )
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = Pinecone.from_texts(
    #     texts=text_chunks, embedding=embeddings, index_name=os.environ['zoft_index_name'])
    # print("vectorstore", vectorstore)
    vectorstore2 = Pinecone.from_existing_index(
        st.secrets["zoft_index_name"], embeddings)
    return vectorstore2

def databasecollection():
    client = pymongo.MongoClient(
        "mongodb+srv://akash:akash@cluster0.rn3if.mongodb.net/?retryWrites=true&w=majority")
    db = client["Zoftwarehub"]
    collection = db["VectorSearch"]

#  Fetch all data from the collection
    all_data = list(collection.find({}))

    for item in all_data:
        print("item", item)
        vectortext = item['vector']
        chunks = get_text_chunks(vectortext)
        vectorstore = get_pinconevectorstore(chunks)
        # print("vectorstore",vectorstore)

# Optionally, you can filter or sort the data before processing
# all_data = list(collection.find({"field": "value"}).sort("field_name"))

# Close the connection
    client.close()
    return all_data

def main():
    load_dotenv()
    # vector_text = databasecollection()

    st.set_page_config(page_title="ZoftBoT",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ZoftBot :books:")
    # print(vectorstore)
    # create conversation chain
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(api_key=openai_api_key)
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  
    user_input = get_text()
    if user_input:
        vectorstore = get_pinconevectorstore(user_input)
        output = Conversation.run(input=vectorstore)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        # output = Conversation.run(input=user_input)  
        # st.session_state.past.append(user_input)
        # st.session_state.generated.append(output)

    with st.expander("Conversation", expanded=True):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.info(st.session_state["past"][i],icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")



if __name__ == '__main__':
    main()