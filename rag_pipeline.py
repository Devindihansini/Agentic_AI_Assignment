# rag_pipeline.py - RAG Implementation

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader

class RAGPipeline:
    """RAG Pipeline for retrieving relevant study materials"""

    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = self._initialize_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    def _initialize_vector_store(self):
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._create_vector_store()

    def _create_vector_store(self):
        sample_docs = [
            "ගණිතය: වර්ගඵලය = දිග × පළල. චතුරස්‍රයක වර්ගඵලය = පාදය²",
            "විද්‍යාව: ආලෝකය තරංග ආකාරයෙන් ගමන් කරයි. ආලෝකයේ වේගය = 3×10⁸ m/s",
            "ඉතිහාසය: සිංහලේ අවසන් රජු වූයේ ශ්‍රී වික්‍රම රාජසිංහ රජතුමාය.",
            "භාෂාව: නාම පද, ක්‍රියා පද, නිපාත පද යනු ව්‍යාකරණ කොටස් වේ.",
        ]
        vector_store = Chroma.from_texts(
            texts=sample_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vector_store.persist()
        return vector_store

    def add_documents(self, file_paths: List[str]):
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)
        self.vector_store.persist()

    def retrieve(self, query: str, subject: str = None, k: int = 3) -> str:
        enhanced_query = f"{subject}: {query}" if subject else query
        results = self.vector_store.similarity_search(enhanced_query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context if context else "අදාළ තොරතුරු හමු නොවීය."

    def evaluate_retrieval(self, test_queries: List[str]):
        evaluations = []
        for query in test_queries:
            context = self.retrieve(query)
            relevance = "Relevant" if len(context) > 10 else "Needs improvement"
            evaluations.append({
                "query": query,
                "context_length": len(context),
                "relevance": relevance
            })
        return evaluations
