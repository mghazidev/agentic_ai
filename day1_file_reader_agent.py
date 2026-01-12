from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

# =========================
# 1️⃣ LOAD LLM (Brain)
# =========================
llm = OllamaLLM(model="llama3")

# =========================
# 2️⃣ LOAD FILE
# =========================
loader = TextLoader("restaurants.txt")
documents = loader.load()
restaurant_data = documents[0].page_content

# =========================
# 3️⃣ PROMPT TEMPLATE
# =========================
prompt = PromptTemplate(
    input_variables=["data", "question"],
    template="""
You are a food data analyst.
Here is restaurant data:
{data}

Answer the following question clearly and accurately:
{question}
"""
)

# =========================
# 4️⃣ CHAIN (Modern LCEL Style)
# =========================
chain = prompt | llm

# =========================
# 5️⃣ ASK QUESTIONS
# =========================
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break
    
    response = chain.invoke({
        "data": restaurant_data,
        "question": question
    })
    print("\n🍽️ Answer:\n", response)