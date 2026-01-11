from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6)

prompt_template = PromptTemplate.from_template(
        """You are a helpful AI bot. Your name is {name}. 
        Ensure you respond in the funny and sarcastic manner.
        Ensure you mention your name in the response.
        {user_input}"""
    )

# example name and user input pairs

name_user_input_pairs = [
    ("John", "Hello, how are you doing?"),
    ("Jane", "I'm doing well, thanks!"),
    ("Jim", "What is the weather in Bangalore?"),
]


for name, user_input in name_user_input_pairs:
    #prepare prompt
    template_object = prompt_template.invoke({"name": name, "user_input": user_input})
    print("formatted prompt: ", template_object) # output is String
    print("type of formatted prompt: ", type(template_object)) # output is StringPromptValue
    print("--------------------------------")
    #invoke the model
    response = llm_openai.invoke(template_object)
    print("response: ", response.content)
    print("--------------------------------")