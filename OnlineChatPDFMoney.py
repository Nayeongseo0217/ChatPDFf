# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''이 파일은 streamlit(온라인)에서만 작동되는 파일입니다. 로컬에서는 작동되지 않습니다.'''
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()

st.title('ChatPDF')
st.write('---')

# 로그 설정
logging.basicConfig(level=logging.INFO)

# API 키 설정
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API Key가 설정되지 않았습니다. .env 파일을 확인하세요.")
else:
    genai.configure(api_key=api_key)

# 파일 업로드
uploaded_file = st.file_uploader('PDF 파일을 업로드하세요')
st.write('---')

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드된 파일이 있을 때
if uploaded_file is not None:
    try:
        pages = pdf_to_document(uploaded_file)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 한 청크당 글자 수
            chunk_overlap=20,  # 청크 간 겹치는 글자 수
            length_function=len,  # 글자 길이를 계산하는 함수
        )

        go = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
        texts = text_splitter.split_documents(pages)
        texts1 = [doc.page_content for doc in texts]

        vectorstore = FAISS.from_texts(texts1, embedding=go)
        retriever = vectorstore.as_retriever()

        template = """Answer the question in sentences based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatGoogleGenerativeAI(model="gemini-pro")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        st.header('ChatPDF에게 질문해보세요!!')
        question = st.text_input('질문을 입력하세요')
        if st.button('질문하기'):
            with st.spinner('답변하는 중...'):
                answer = chain.invoke(question)
                st.write(answer)
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
