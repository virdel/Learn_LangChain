
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp


## model 
llm = LlamaCpp(model_path="./models/dolphin-2.2.1-mistral-7b.Q2_K.gguf",temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print(conversation.predict(input="hi, my name is lang!"))

print(memory.buffer)
# print(conversation.predict(input="What is 1+1?"))
# print(conversation.predict(input="What is my name?"))