import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone
import os
import pymongo
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.chat_models.base import BaseChatModel


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
        api_key=os.environ['zoft_pincone_API_KEY'],  # find at app.pinecone.io
        # next to api key in console
        environment=os.environ['zoft_pincone_env'],
    )
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = Pinecone.from_texts(
    #     texts=text_chunks, embedding=embeddings, index_name=os.environ['zoft_index_name'])
    # print("vectorstore", vectorstore)
    vectorstore2 = Pinecone.from_existing_index(
        os.environ['zoft_index_name'], embeddings)
    return vectorstore2


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = get_conversation_chain_without_verbose(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_conversation_chain_without_verbose(llm, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )


def handle_userinput(user_question):
    print("handle_userinput", user_question)
    vectorstore = get_pinconevectorstore(user_question)
    # print(vectorstore)
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)
    response = st.session_state.conversation({'question': user_question})
    # print("response", response)

    conversation_item = {"question": user_question, "answer": response['answer']}
    st.session_state.chat_history.append(conversation_item)
        
    # Display chat history
    st.header("Chat History")
    if hasattr(st.session_state, "chat_history"):
        for item in st.session_state.chat_history:
            st.write("You:", item["question"])
            st.write("ZoftBot:", item["answer"])
            st.write("")



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
        st.session_state.chat_history = []

    st.header("ZoftBot :books:")
    user_question = st.text_input("Ask your Doubt:")
    if user_question:
        handle_userinput(user_question)
        # for i, message in enumerate(st.session_state.chat_history):
        #     if i % 2 == 0:
        #         st.write(user_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)
        #     else:
        #         st.write(bot_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)

    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)

    #             # get the text chunks
    #             text_chunks = get_text_chunks(raw_text)

    #             # create vector store
        # vectorstore = get_pinconevectorstore(text_chunks)
    #             print(vectorstore)
    #             # create conversation chain
    #             st.session_state.conversation = get_conversation_chain(
    #                 vectorstore)


if __name__ == '__main__':
    main()
