from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

st.title("AI Chat Bot")
st.markdown("my qna bot with langchain and google gemini")

if "messages" not in st.session_state:
   st.session_state.messages=[]

for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("ask anything ?")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res=llm.invoke(query)
    st.chat_message("ai").markdown(res.content[0]["text"])
    st.session_state.messages.append({"role":"ai","content":res.content[0]["text"]})
    
   

#You type a message and press Enter.
#st.chat_input() detects the input.
#Streamlit reruns your script from the top.
#st.session_state preserves your chat history.
#The for loop displays all previous messages
#The if query: block executes and sends your new question to Gemini.
#Gemini's response is displayed and saved.