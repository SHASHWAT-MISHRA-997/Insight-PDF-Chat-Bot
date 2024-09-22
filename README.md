# 📚 InsightBot: Intelligent Document Interaction

Welcome to InsightBot an AI-powered tool that allows users to upload and interact with PDF documents through an intelligent chatbot. This application uses models like OpenAI GPT for answering questions based on the content of your PDFs and also offers additional analytical functionalities like word count, character count and sentiment analysis.

# 🚀 Features

Upload PDFs: Users can upload one or more PDF documents to the app.
Ask Questions: After uploading PDFs, users can ask questions related to the content of the documents.
API Key Support: Supports OpenAI's GPT models for text generation.
PDF Analytics: Enables word count, character count and sentiment analysis using VADER.
Chat History: Displays the conversation history in an intuitive format.

# 📂 Demo Video
[Demo video link](https://www.linkedin.com/posts/sm980_excited-to-share-my-another-project-based-activity-7243682941816520704-IQwY?utm_source=share&utm_medium=member_desktop)

# 📂 Image : 
[Image](https://www.linkedin.com/feed/update/urn:li:ugcPost:7243682781694726144?commentUrn=urn%3Ali%3Acomment%3A%28ugcPost%3A7243682781694726144%2C7243689240503439360%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287243689240503439360%2Curn%3Ali%3AugcPost%3A7243682781694726144%29)
# 🛠️ Technologies Used

Python
Streamlit for the web interface.
OpenAI API for GPT models.
PyPDF2 for PDF text extraction.
VADER Sentiment Analysis for sentiment analysis.
Hugging Face Models (optional, to be added later).
Werkzeug for secure file handling.

# 📝 Installation
Prerequisites:
Python 3.7+
A valid OpenAI API key (if using OpenAI models)
Installation Steps:
Clone the Repository:

bash
Copy code
git clone https://github.com/Shashwat876/InsightBot.git
cd InsightBot
Create a Virtual Environment (Optional but recommended):

bash
Copy code
python -m venv env
source env/bin/activate    # On Windows: env\Scripts\activate
Install Required Libraries:

bash
Copy code
pip install -r requirements.txt
Set Up API Keys:

Obtain your OpenAI API key from here.
Add it directly in the application or in the sidebar when running the app.
🎮 Usage
Run the Application:

bash
Copy code
streamlit run app.py
Upload PDF Documents:

Go to the Your Documents section in the sidebar.
Upload one or more PDF files.
Enable PDF Analytics:

Navigate to PDF Analytics Settings.
Enable options like Word Count, Character Count and Sentiment Analysis.
Ask Questions:

Type your question in the input box and click "Submit".
The chatbot will provide an answer based on the uploaded PDFs.
View the Conversation:


# 🛡️ Security
Ensure that your OpenAI API key remains confidential and avoid sharing it publicly.

# 🤝 Contributions
Contributions, issues and feature requests are welcome! Feel free to check the issues page to report any issues or request new features.

# 🧑‍💻 Author
Shashwat Mishra
[LinkedIn](https://www.linkedin.com/in/sm980/)

# 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

