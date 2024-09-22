import os  
import streamlit as st
from werkzeug.utils import secure_filename
import openai
from prompts import set_prompt
from textFunctions import get_text_chunks
from vizFunctions import vaders_barchart
from htmlTemplates import css
from PyPDF2 import PdfReader

# Constants
MAX_TOKENS = 1024  # Max token limit for input to avoid truncation issues

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
        "model_choice": "OpenAI",  # Default model choice
        "openai_api_key": "",
        "huggingface_api_key": "",
        "huggingface_model_choice": "distilbert-base-uncased"  # Default Hugging Face model
    }
    for state, default_value in session_states.items():
        if state not in st.session_state:
            st.session_state[state] = default_value

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
    openai_key = st.session_state.openai_api_key
    if not openai_key and not st.session_state.huggingface_api_key:
        st.error("⚠️ Please enter an API key for OpenAI or Hugging Face.")
        return False
    return True

def handle_userinput(user_question):
    try:
        if not st.session_state.pdf_content:
            st.error("❌ No PDF content available.")
            return
        
        model_choice = st.session_state.model_choice

        if model_choice == "OpenAI":
            openai.api_key = st.session_state.openai_api_key
            prompt = f"Context: {st.session_state.pdf_content}\n\nQuestion: {user_question}\nAnswer:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4"
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message['content'].strip()

        elif model_choice == "Hugging Face":
            # Placeholder for Hugging Face model interaction
            answer = "Hugging Face model response here."  # Replace with actual Hugging Face model call

        else:
            st.error("❌ Invalid model selection.")
            return

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
        st.session_state.pdf_processed = True
        st.success("✅ PDFs processed.")

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        st.session_state.pdf_processed = False

def display_convo(prompt):
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)

def chatbot_settings():
    with st.expander("🤖 Chat Bot Settings", expanded=True):
        st.session_state.personality = st.selectbox(label='Personality', options=['general assistant', 'academic', 'witty'])
        st.session_state.temp = st.slider("Temperature", 0.0, 1.0, 0.5)

        
        st.text_input("OpenAI API Key", type="password", key="openai_api_key", placeholder="Enter your OpenAI API key")
        st.markdown("[Get OpenAI API Key](https://platform.openai.com/signup/)")

      
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
    with st.sidebar:
        st.header("🔑 API Keys")
        
        chatbot_settings()
        
        # PDF Analytics Settings
        pdf_analytics_settings()

        # PDF Documents Section
        with st.expander("📂 Your Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")

        if st.button("🚀 Process Files + New Chat"):
            if validate_api_keys():
                if pdf_docs:
                    with st.spinner("⏳ Processing..."):
                        process_docs(pdf_docs)
                        # Call PDF analytics after processing
                        pdf_analytics(pdf_docs)
                else:
                    st.error("⚠️ Please upload at least one PDF file.")

def main():
    st.set_page_config(page_title="📚 InsightBot: Intelligent Document Interaction", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)
    init_ses_states()

    st.title("📚 Document-Insight-Bot")
    st.markdown("""
        <hr style="border: 1px solid gray;"/>
        <footer>
            <p style="text-align: center;">
                Developed by <a href="https://www.linkedin.com/in/sm980/" target="_blank">SHASHWAT MISHRA</a>
            </p>
        </footer>
    """, unsafe_allow_html=True)
    st.markdown("""
        Welcome to **InsightBot**! 🎉 Here you can upload your PDF documents and interact with their content using our intelligent chatbot.

        **Instructions:**
        1. **API Keys** 🔑:
            - Enter your API keys for OpenAI in the sidebar.
            - [Get OpenAI API Key](https://platform.openai.com/signup/)
            

        2. **Upload PDF Documents** 📂:
            - Navigate to the **Your Documents** section in the sidebar.
            - Upload one or more PDF files.

        3. **Enable PDF Analytics** 📊:
            - Go to the **PDF Analytics Settings** and enable analytics options (e.g., Word Count, Character Count).

        4. **Ask Questions** ❓:
            - Type your question about the uploaded PDFs in the text box and hit **Submit**. 

        5. **View Responses** 📝:
            - Your conversation will be displayed below the input box.

        **Note**: Make sure to keep your API keys secure and not share them publicly! 🚨
    """)

    sidebar()

    if st.session_state.pdf_processed:
        user_question = st.text_input("🔍 Ask a question about your PDFs:", "")
        if st.button("🗨️ Submit"):
            handle_userinput(user_question)

        if st.session_state.chat_history:
            display_convo(user_question)


if __name__ == "__main__":
    main()
