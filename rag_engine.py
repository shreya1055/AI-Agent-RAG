"""
rag_engine.py — The RAG (Retrieval-Augmented Generation) Engine
Location: rag-agent-project/rag_engine.py

This is the core of the project. It handles:
  1. Loading .txt documents from a folder
  2. Splitting them into bite-sized chunks
  3. Converting chunks into vector embeddings (via Gemini)
  4. Storing those vectors in ChromaDB
  5. Searching ChromaDB for relevant chunks given a question
  6. Sending those chunks + the question to Gemini to generate an answer

Two phases:
  INDEXING  (run once)   →  documents  →  chunks  →  vectors  →  ChromaDB
  QUERYING  (every ask)  →  question   →  vector  →  search   →  LLM answer
"""
import os

from dotenv import load_dotenv
# TextLoader used for reading a single .txt file from disk and returns a LangChain Document object
# DirectoryLoader can find matching files in the folder and load them for you.
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GoogleGenerativeAIEmbeddings: Wraps Google's embedding API. It implements LangChain's Embeddings interface, which means any
# LangChain component that needs embeddings (like ChromaDB) can use it interchangeably with OpenAI, Cohere, etc.
#
# ChatGoogleGenerativeAI: wraps Google's Gemini chat API. It implements LangChain's BaseChatModel interface, so any
# LangChain chain can use it as a drop-in replacement for ChatOpenAI, ChatAnthropic, etc.
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Chroma: LangChain's wrapper around ChromaDB
from langchain_chroma import Chroma

# create_stuff_documents_chain: Combines an LLM + a prompt into a chain that takes a list of Documents as "context"
# and produces a text answer. "stuff" = concatenate all documents into the prompt.
from langchain.chains.combine_documents import create_stuff_documents_chain

# create_retrieval_chain: Wraps a retriever + a document chain into a single chain. When you call .invoke(), it:
# (1) Sends the question to the retriever → gets documents, (2) Passes those documents to the document chain → answer,
# (3) Returns {"input": ..., "context": [...], "answer": ...}
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate  # Uses structured list of messages to build prompt (not str)

load_dotenv()

class RAGEngine:
    """
    Encapsulates the entire RAG pipeline.

    This class is a "service object" — you create one instance and
    reuse it for the lifetime of the server. It holds:
      - The embedding model (for converting text <-> vectors)
      - The LLM (for generating answers)
      - The vector store (for storing and searching chunks)

    Typical usage:
        engine = RAGEngine()
        engine.ingest_documents("./documents")   # once
        result = engine.query("What is FastAPI?") # many times
    """
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "my_documents"):
        """
        Set up the embedding model, LLM, and prepare for ChromaDB.
        Parameters:
        - persist_directory: The folder where ChromaDB saves its data. This folder is auto-created on first ingest.
        - collection_name: A name for the collection inside ChromaDB. One ChromaDB instance can hold multiple collections,
          like tables in a SQL database. Each collection has its own set of documents and its own vector index.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # Google's text-embedding-004 converts text into a 768-dimensional vector
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
        self.vector_store = None

    def ingest_documents(self, documents_path: str) -> dict:
        """
        Read documents → split into chunks → embed → store in ChromaDB. Indexing step of RAG.

        - Parameter: documents_path: Path to folder with .txt files.
        - Returns: dict with {"documents_loaded": int, "chunks_created": int, "status": str}
        """

        # STEP 1: LOAD
        # DirectoryLoader walks the folder and applies TextLoader to every file matching the glob pattern.
        # glob="**/*.txt" checks for any depth of subdirectories ending in .txt
        loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No .txt documents found in {documents_path}")

        # STEP 2: SPLIT
        # chunk_size=500: maximum characters per chunk.
        # chunk_overlap=50: consecutive chunks share 50 characters.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
        # Each chunk is still a Document, but with shorter page_content and the same metadata (source file) as its parent.
        chunks = text_splitter.split_documents(documents)

        # STEP 3: EMBED AND STORE
        # Chroma.from_documents() does Embeds, Stores (inserts vector, text, meta data) into DB, and Persists. ChromaDB builds
        # w/ HNSW index (Hierarchical Navigable Small World), which is a graph-based indexing algorithm that
        # builds multi-layered proximity graphs that let the system skip large portions of the dataset during a search
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.persist_directory, collection_name=self.collection_name)

        return {"documents_loaded": len(documents), "chunks_created": len(chunks), "status": "success",}

    def load_existing_store(self) -> bool:
        """
        Load a previously persisted ChromaDB store from disk. Call this at server startup. If documents were ingested in a
        previous run, ChromaDB reads the SQLite + HNSW files from persist_directory and reconstructs the in-memory index.
        This means you don't have to re-embed everything on restart.

        Returns: True if the store was loaded, False if the folder doesn't exist.
        """
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings, collection_name=self.collection_name)
            return True
        return False

    def query(self, question: str, k: int = 3) -> dict:
        """
        Answer a question using RAG: retrieve context, then generate.

        The pipeline for each query:
        (1) Q. -> (GoogleGenerativeAIEmbeddings) -> ChromaDB search ->
        (2) Get top k chunks scored (e.g. 0.92, 0.88) ->
        (3) ChatPromptTemplate ->
        (4) Gemini 2.5 Flash
        (5) Output result

        Parameters:
        - question: The user's natural language question.
        - k: Number of chunks to retrieve. Consider tradeoff of more context vs.
             more token usage. chunks=3 is a good default.
        Returns: Dict with {"question", "answer", "sources"}.
        """
        if not self.vector_store:
            raise ValueError("No documents ingested yet. Call ingest_documents() first.")

        # ChatPromptTemplate creates a structured conversation:
        #   - ("system", "..."): sets the AI's behavior/role
        #   - ("human", "..."):  the user's message
        # Placeholders:
        # - {context}: the retrieval chain fills this with the text of the retrieved document chunks, joined together.
        # - {input}: the retrieval chain fills this with the user's question.
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful AI assistant. Use the following pieces of context to answer the question. If the context 
             doesn't contain enough information to answer, say so honestly — don't make things up."""),
            ("human",
             "Context:\n{context}\n\nQuestion: {input}"),
        ])
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        #   create_retrieval_chain(retriever, document_chain)
        #   Wraps a retriever + the document chain into one pipeline.
        #   When you call .invoke(), it:
        #     1. Sends the question to the retriever
        #     2. The retriever embeds the question → searches ChromaDB
        #        → returns top-k Document objects
        #     3. Passes those Documents to the document chain as context
        #     4. The document chain stuffs them into the prompt → LLM
        #     5. Returns a dict with:
        #          "input":   the original question
        #          "context": list of retrieved Document objects
        #          "answer":  the LLM's generated answer
        #
        #   self.vector_store.as_retriever() creates a retriever that:
        #     1. Embeds the question into a vector
        #     2. Runs cosine similarity search in ChromaDB
        #     3. Returns the top-k results
        #   search_kwargs={"k": k} sets how many chunks to retrieve.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # RUN THE CHAIN
        # .invoke() triggers the entire pipeline:
        #   1. Embed the question → query vector
        #   2. Search ChromaDB → top-k chunks
        #   3. Fill prompt template → system + human messages
        #   4. Send to Gemini → generated answer
        #   5. Return {"input": "...", "context": [...], "answer": "..."}
        result = retrieval_chain.invoke({"input": question})

        # Formatted response
        sources = []
        for doc in result.get("context", []):
            sources.append({"content": doc.page_content, "metadata": doc.metadata})

        return {"question": question, "answer": result["answer"], "sources": sources}



