import streamlit as st
import os

os.environ["OPENAI_API_KEY"]="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"
from langchain_community.document_loaders import TextLoader
#문서 준비 및 전처리
loader = TextLoader("국립한밭대학교 학칙(제59차 일부개정 2025.4.15.txt")
docs = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter=CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
split_docs = text_splitter.split_documents(docs)

#문서 임베딩
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#FAISS에 벡터 저장(인덱싱)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    return_source_documents=True
)

st.set_page_config(page_title="한밭대 학칙 챗봇", layout="centered")
st.title("한밭대학교 학칙 챗봇")
st.caption("OpenAI 기반 RAG 방식 챗봇")
st.image("한밭대로고.png", width=200)
st.markdown("---")
st.markdown("### 아래 입력창에 궁금한 학칙을 작성해주세요")
st.markdown("ex)휴학 조건이 어떻게 되나요?")
st.markdown("---")

question = st.text_input("궁금한 내용을 입력하세요:")
answer = None

if st.button("검색") and question:
   with st.spinner("답변 생성중..."):
      answer = qa({"query": question})

   if answer:
        st.success("답변 생성 완료")
        st.markdown("### GPT 답변:")
        st.write(answer["result"])

        with st.expander("참고한 학칙 문서"):
          for i, doc in enumerate(answer["source_documents"]):
              st.markdown(f"**문서 {i+1}:**")
              st.write(doc.page_content[:200])
   else:   
       st.error("답변 생성 실패 다시 시도해주세요..")
elif question:
    st.info("질문을 입력한 후 '검색'버튼을 눌러 주세요")
else:
    st.warning("질문을 입력해주세요.")
