from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# =========================
# 1️⃣ LOAD FILE
# =========================
loader = TextLoader("restaurants.txt")
documents = loader.load()

# =========================
# 2️⃣ SPLIT INTO CHUNKS
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# =========================
# 3️⃣ CREATE EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(model="llama3")

# =========================
# 4️⃣ VECTOR STORE (MEMORY)
# =========================
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# 5️⃣ LOAD LLM (BRAIN)
# =========================
llm = OllamaLLM(model="llama3")

# =========================
# 6️⃣ PROMPT
# =========================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a restaurant data analyst.

Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

# =========================
# 7️⃣ RAG LOOP
# =========================
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    print("\n🍽️ Answer:\n", response)
  