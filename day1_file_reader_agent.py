from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# =========================
# 1️⃣ LOAD LLM (Brain)
# =========================
llm = Ollama(model="llama3")

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
# 4️⃣ CHAIN (Glue)
# =========================
chain = LLMChain(llm=llm, prompt=prompt)

# =========================
# 5️⃣ ASK QUESTIONS
# =========================
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    response = chain.run(
        data=restaurant_data,
        question=question
    )

    print("\n🍽️ Answer:\n", response)
