from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import SimpleSequentialChain
llm = LlamaCpp(model_path="./models/dolphin-2.2.1-mistral-7b.Q2_K.gguf",temperature=0.0)

# prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe \
#     a company that makes {product}?"
# )


# chain = LLMChain(llm=llm, prompt=prompt)

# product = "Queen Size Sheet Set"
# print(chain.run(product))

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

product = "Queen Size Sheet Set"
print(overall_simple_chain.run(product))