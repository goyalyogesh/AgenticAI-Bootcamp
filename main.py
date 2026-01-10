from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def askGoogle(m1):
    response = llm_google.invoke(m1)
    return response.content

def askOpenAI(m1):
    response = llm_openai.invoke(m1)
    return response.content

# http://127.0.0.1:7860
demo = gr.Interface(
    fn=askGoogle,
    inputs=gr.Textbox(label="Ask Your question here"),
    outputs=gr.Textbox(label="Response"),
    title="Chat with Gemini",
    description="Chat with Google's Gemini model"
)

if __name__ == "__main__":
    demo.launch()