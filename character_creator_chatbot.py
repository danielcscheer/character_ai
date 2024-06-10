import streamlit as st
from decouple import config
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

INIT_MESSAGE = {
        "role": "assistant",
        "content": f"""What do you want to talk about?"""
        }

# Set Streamlit page configuration
st.set_page_config(page_title='🤖 Create Your Own AI Character 🤖', layout='wide')
st.title("🤖 Create Your Own AI Character 🤖")

# Sidebar info
with st.sidebar:
    st.markdown("## Create Your Character")
    # define character attributes
    CHARACTER_NAME = st.text_input("What is your character's name", value="")
    CHARACTER_TYPE = st.text_input("What is your character (e.g. a lion, a robot, etc.)", value="")
    CHARACTER_EMOTIONS = st.text_input("What types of emotions does your character have (e.g. happy, mean, sad, etc.)", value="")
    CHARACTER_DESCRIPTION = st.text_area("Describe your character (provide a back story, life details, etc.)", value="")

    st.markdown("## Model Settings")
    TEMPERATURE = st.slider("Temperature", min_value=0.0,
                            max_value=1.0, value=0.1, step=0.1)
    TOP_P = st.slider("Top-P", min_value=0.0,
                      max_value=1.0, value=0.9, step=0.01)
    TOP_K = st.slider("Top-K", min_value=1,
                      max_value=500, value=10, step=5)
    MAX_TOKENS = st.slider("Max Token", min_value=0,
                           max_value=2048, value=1024, step=8)
    MEMORY_WINDOW = st.slider("Memory Window", min_value=0,
                              max_value=10, value=3, step=1)

# Initialize the ConversationChain
def init_conversationchain() -> ConversationChain:

    ANTHROPIC_API_KEY=config('ANTHROPIC_KEY')

    CHARACTER_INFO = f"""
    The following is a conversation between a human and an AI character. The AI charcter is talkative and provides lots of specific details from its context:
    You are a {CHARACTER_TYPE} named {CHARACTER_NAME}. You are {CHARACTER_EMOTIONS}. 
    {CHARACTER_DESCRIPTION}.

    - Always stay in character as {CHARACTER_NAME} the {CHARACTER_TYPE}
    - If you are unsure how to respond, respond with another question.
    \n\n
    """

    CHARACTER_PROMPT = CHARACTER_INFO + "/n/n" + """
    Current conversation:
    {history}
    Human: {input}
    AI Character:
    """  
    
    DEFAULT_INPUT_VARIABLES = ["history", "input"]
    CHARACTER_PROMPT = PromptTemplate(input_variables=DEFAULT_INPUT_VARIABLES, template=CHARACTER_PROMPT)
    
    llm = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        model="claude-3-haiku-20240307",
        max_tokens_to_sample=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        streaming=True     
        )

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=MEMORY_WINDOW, ai_prefix="Character", chat_memory=StreamlitChatMessageHistory()),
        prompt=CHARACTER_PROMPT
    )

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [INIT_MESSAGE]
    return conversation

def generate_response(conversation, input_text):
    return conversation.run(input=input_text, callbacks=[StreamHandler(st.empty())])


# Re-initialize the chat
def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    conv_chain = init_conversationchain()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Initialize the chat
conv_chain = init_conversationchain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User-provided prompt
prompt = st.chat_input()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # print(st.session_state.messages)
        response = generate_response(conv_chain, prompt)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)