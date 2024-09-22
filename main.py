import os
import streamlit as st
from werkzeug.utils import secure_filename
from transformers import pipeline, GPT2Tokenizer
from prompts import set_prompt
from textFunctions import get_text_chunks
from vizFunctions import vaders_barchart
from htmlTemplates import css
from PyPDF2 import PdfReader

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# Constants
MAX_TOKENS = 1024

# Initialize Hugging Face pipeline for default usage
hf_conversation_chain = pipeline("text-generation", model="gpt2-large", max_length=MAX_TOKENS, temperature=0.5)

def get_hf_conversation_chain():
    def conversation_chain(prompt):
        response = hf_conversation_chain(prompt, return_full_text=False)
        return response[0]['generated_text'].strip()
    return conversation_chain

def init_ses_states():
    session_states = {
        "conversation": None,
        "chat_history": [],
        "pdf_analytics_enabled": False,
        "display_char_count": False,
        "display_word_count": False,
        "display_vaders": False,
        "pdf_processed": False,
        "pdf_content": "",
        "api_choice": "Cohere",  # Default API choice
        "openai_api_key": "",     # Added OpenAI API key to session state
        "cohere_api_key": ""      # Added Cohere API key to session state
    }
    for state, default_value in session_states.items():
        if state not in st.session_state:
            st.session_state[state] = default_value

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def get_pdf_text(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def get_pdfs_text(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        pdf_text = get_pdf_text(pdf_file)
        if pdf_text.strip():
            all_text += pdf_text + "\n"
    return all_text

def validate_api_keys():
    cohere_key = st.session_state.cohere_api_key
    openai_key = st.session_state.openai_api_key
    if st.session_state.api_choice == "Cohere" and not cohere_key:
        st.error("⚠️ Please enter your Cohere API key.")
        return False
    elif st.session_state.api_choice == "OpenAI" and not openai_key:
        st.error("⚠️ Please enter your OpenAI API key.")
        return False
    return True

def handle_userinput(user_question):
    try:
        if not st.session_state.pdf_content:
            st.error("❌ No PDF content available.")
            return
        
        response = qa_pipeline(question=user_question, context=st.session_state.pdf_content)
        answer = response["answer"]

        st.session_state.chat_history.append({'content': answer})
        display_convo(user_question)

    except Exception as e:
        st.error(f"Error handling user input: {e}")

def process_docs(pdf_docs):
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = []
    st.session_state["user_question"] = ""

    try:
        raw_text = get_pdfs_text(pdf_docs)
        if not raw_text.strip():
            st.error("⚠️ No text content extracted from the uploaded PDF(s). Please check the file.")
            return

        st.session_state.pdf_content = raw_text
        st.session_state.conversation = get_hf_conversation_chain()
        
        if st.session_state.conversation:
            st.session_state.pdf_processed = True
            st.success("✅ PDFs processed and conversation chain initialized.")
            pdf_analytics(pdf_docs)
        else:
            st.session_state.pdf_processed = False
            st.error("❌ Failed to create conversation chain.")
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        st.session_state.pdf_processed = False

def display_convo(prompt):
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble user-bubble">{message["content"][len(prompt):]}</div>', unsafe_allow_html=True)

def pdf_analytics(pdf_docs):
    all_text = ""
    if st.session_state.pdf_analytics_enabled:
        with st.container():
            for pdf in pdf_docs:
                st.subheader(str(secure_filename(pdf.name)))
                text = get_pdf_text(pdf)
                all_text += text

                if st.session_state.display_word_count:
                    st.markdown(f'<p class="small-font"># of Words: {len(text.split())}</p>', unsafe_allow_html=True)

                if st.session_state.display_char_count:
                    st.markdown(f'<p class="small-font"># of Characters: {len(text)}</p>', unsafe_allow_html=True)

                if st.session_state.display_vaders:
                    vaders_barchart(text, name=str(secure_filename(pdf.name)))

            if len(pdf_docs) > 1:
                if any([st.session_state.display_word_count, st.session_state.display_char_count, st.session_state.display_vaders]):
                    st.subheader("Collective Summary:")
                    if st.session_state.display_word_count:
                        st.markdown(f'<p class="small-font"># of Words: {len(all_text.split())}</p>', unsafe_allow_html=True)

                    if st.session_state.display_char_count:
                        st.markdown(f'<p class="small-font"># of Characters: {len(all_text)}</p>', unsafe_allow_html=True)

                    if st.session_state.display_vaders:
                        vaders_barchart(all_text, name=str(secure_filename(pdf_docs[-1].name)))

def chatbot_settings():
    with st.expander("🤖 Chat Bot Settings", expanded=True):
        st.session_state.personality = st.selectbox(label='Personality', options=['general assistant', 'academic', 'witty'])
        st.session_state.temp = st.slider("Temperature", 0.0, 1.0, 0.5)

def pdf_analytics_settings():
    with st.expander("📈 PDF Analytics Settings", expanded=True):
        enable_pdf_analytics = st.checkbox("Enable PDF Analytics")
        st.session_state.pdf_analytics_enabled = enable_pdf_analytics
        if enable_pdf_analytics:
            st.caption("✅ PDF Analytics Enabled")
            st.caption("🔧 Display Options")
            st.session_state.display_char_count = st.checkbox("Character Count")
            st.session_state.display_word_count = st.checkbox("Word Count")
            st.session_state.display_vaders = st.checkbox("VADER Sentiment Analysis")
        else:
            st.caption("❌ PDF Analytics Disabled")

def sidebar():
    global pdf_docs
    with st.sidebar:
        st.header("🔑 API Keys")
        
        st.selectbox("Select API", options=["Cohere", "OpenAI"], key="api_choice")
        
        if st.session_state.api_choice == "Cohere":
            st.text_input("Cohere API Key", type="password", key="cohere_api_key", placeholder="Enter your Cohere API key")
            st.markdown("[Get Cohere API Key](https://cohere.ai/get-started)")
        elif st.session_state.api_choice == "OpenAI":
            st.text_input("OpenAI API Key", type="password", key="openai_api_key", placeholder="Enter your OpenAI API key")
            st.markdown("[Get OpenAI API Key](https://platform.openai.com/signup)")

        chatbot_settings()
        pdf_analytics_settings()
        
        # PDF Documents Section
        with st.expander("📂 Your Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
            if st.button("🚀 Process Files + New Chat"):
                if validate_api_keys():
                    if pdf_docs:
                        with st.spinner("⏳ Processing..."):
                            process_docs(pdf_docs)
                    else:
                        st.caption("⚠️ Please Upload At Least 1 PDF")
                        st.session_state.pdf_processed = False

def main():
    st.set_page_config(page_title="📚 InsightBot: Intelligent Document Interaction", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("""
    <div class="content text-center py-8">
        <h1 class="text-4xl font-extrabold text-green-800 mb-6 bg-clip-text text-transparent bg-gradient-to-r from-green-500 to-green-600">
            📚 Multi-Document Chat Bot
        </h1>
        <p class="text-lg text-gray-800 max-w-3xl mx-auto leading-relaxed bg-gray-100 p-6 rounded-lg shadow-md">
            Welcome to the Multi-Document Chat Bot! Interact seamlessly with your PDF documents. 
            Just upload your PDFs and ask questions related to the content! 😊
        </p>
        <p class="text-sm text-gray-600 max-w-3xl mx-auto leading-relaxed bg-gray-100 p-6 rounded-lg shadow-md">
            ✨ Instructions: <br>
            1. Upload your PDF files on the sidebar. 📂 <br>
            2. Enter your API keys for Cohere, Pinecone, and OpenAI. 🔑 <br>
            3. Enable PDF Analytics if you want additional insights. 📊 <br>
            4. Type your questions about the content in the text box below. 💬 <br>
            5. Press "Submit" to receive your answers! 🚀 <br>
            6. Enjoy exploring your documents! 🎉
        </p>
    </div>
    """, unsafe_allow_html=True)

    init_ses_states()
    sidebar()
    if st.session_state.pdf_processed:
        user_question = st.text_input("💬 Type your question here...")
        if st.button("Submit", key="submit_button"):
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
