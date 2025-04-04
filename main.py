import os
import re
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class BookProcessor:
    def __init__(self, model_name="gpt-4o"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300, 
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        )

    def extract_text(self, pdf_path):
        with fitz.open(pdf_path) as doc:
            text = "".join([page.get_text() for page in doc])
            return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    def process_documents(self, text, persist_dir):
        document = Document(page_content=text)
        docs = self.text_splitter.split_documents([document])
        return Chroma.from_documents(docs, self.embeddings, persist_directory=persist_dir)

class BookQA:
    def __init__(self, db):
        self.retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
        self.llm = ChatOpenAI(model="gpt-4o")
        
        self.summary_chain = ChatPromptTemplate.from_template(
            "Summarize this text focusing on core thesis:\n{context}"
        ) | self.llm
        
        self.qa_chain = ChatPromptTemplate.from_template(
            "Answer based on context:\n{context}\nQuestion: {query}"
        ) | self.llm

        self.chain = (
            RunnableParallel({
                "query": RunnablePassthrough(),
                "context": self.retriever
            })
            | RunnableBranch(
                (lambda x: "summary" in x["query"].lower(), self.summary_chain),
                self.qa_chain
            )
            | StrOutputParser()
        )


    def query(self, question):
        return self.chain.invoke(question)

if __name__ == "__main__":
    processor = BookProcessor()
    pdf_text = processor.extract_text("kniga.pdf")
    
    persist_dir = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
    db = processor.process_documents(pdf_text, persist_dir)
    
    qa_system = BookQA(db)
    
    print("--- Book Summary ---")
    print(qa_system.query("Summarize the first main topic"))
    
    print("\n--- Question Answer ---")
    print(qa_system.query("What is the central argument presented?"))