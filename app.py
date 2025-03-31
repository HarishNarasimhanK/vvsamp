import streamlit as st
from openai import OpenAI
import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import pypdf
from io import BytesIO
import PyPDF2
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Databricks API token
DATABRICKS_TOKEN = "dapi7ffd306c9120591cb502ca6a737690c8"

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-1225341724046092.12.azuredatabricks.net/serving-endpoints"
)
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:jBQauPYcqCRsLUaNFcPCBZxH:93ccaa69209746e66259b73577948f710cd3327b5f7684391bf8ddb486b3564f"
ASTRA_DB_ID = "41f9c77f-d12c-4d5e-868d-1ae82021da72"
groq_api_key = "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"
model = ChatGroq(model_name = 'llama-3.3-70b-versatile',groq_api_key = groq_api_key)


import os
from groq import Groq

audio_client = Groq(api_key=groq_api_key)


      
# Initialize HuggingFace Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"token": "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"}
    )

# Initialize Astra Vector Index
@st.cache_resource
def get_astra_vector_index():
    with st.spinner("Loading Vector Store... Please wait"):
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        
        astra_vector_store = Cassandra(
            embedding=get_embeddings(),
            table_name="TREDENCE_AI_HACKATHON",
            session=None,
            keyspace=None,
        )

        return VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Cache and use the vector index
astra_vector_index = get_astra_vector_index()

SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "kn": "Kannada",
        "mr": "Marathi",
        "ml": "Malayalam",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ur": "Urdu",
        "as": "Assamese"
}

if "page" not in st.session_state:
    st.session_state.page = "home"
    
# Home page with buttons
def home():
    st.title("Welcome to VidhiVani")
    st.write("Please select an option:")
    
    if(st.session_state.user_type == "lawyer"):
        col1, col2, col3, col4 = st.columns(4, vertical_alignment='bottom')
        with col1:
            if st.button("LEGAL QNA CHATBOT"):
                st.session_state.page = "qna"
                st.rerun()
        with col2:
            if st.button("JUDGEMENT DOCUMENT SUMMARIZER"):
                st.session_state.page = "summarizer"
                st.rerun()
        with col3:
            if st.button("CONTRACT ANALYZER"):
                st.session_state.page = "contract"
                st.rerun()
        with col4:
            if st.button("PRIOR CASE RETRIEVER"):
                st.session_state.page = "PriorCaseRetrieval"
                st.rerun()
    else:
        col1, col2 = st.columns(2, vertical_alignment='bottom')
        with col1:
            if st.button("LEGAL QNA CHATBOT"):
                st.session_state.page = "qna"
                st.rerun()
        with col2:
            if st.button("CONTRACT ANALYZER"):
                st.session_state.page = "contract"
                st.rerun()

def load_qna_history():
    if st.session_state.user_type == "lawyer":
        logname = "lchat_history.pkl"
    else:
        logname = "uchat_history.pkl"
    
    if os.path.exists(logname):
        try:
            with open(logname, 'rb') as f:
                data = pickle.load(f)
                if not data:
                    return []
                return data
        except EOFError:
            return []
    
    try:
        with open(logname, 'x') as f:
            pass
    except FileExistsError:
        pass
    
    return []

def rag():
    with st.container():
        col1, col2 = st.columns([0.9, 0.1], vertical_alignment="bottom")
        with col1:
            st.write("## Prior Case Retriever")
        with col2:
            if st.button("Back", help="Go to Home"):
                st.session_state.page = "home"
                st.rerun()

    def similar_case_retriever(input_text):
        answer = astra_vector_index.query(input_text, llm=model).strip()
        return answer
    

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    msgbox = st.container(border=True, height=480)
    for message in st.session_state.rag_messages:
        msgbox.chat_message(message["role"]).markdown(message["content"])

    # User input
    user_input = st.chat_input("Enter the type of cases you want to retrieve")
    
    if user_input:
        # Append user message
        st.session_state.rag_messages.append({"role": "user", "content": user_input})
        
        # Retrieve response
        with st.spinner("Thinking..."):
            response = similar_case_retriever(user_input)
        
        # Append assistant response
        st.session_state.rag_messages.append({"role": "assistant", "content": response})
        
        # Display chat history
        for message in st.session_state.rag_messages:
            msgbox.chat_message(message["role"]).markdown(message["content"])
        
        # Save chat history
        with open('retr_history.pkl', 'wb') as f:
            pickle.dump(st.session_state.rag_messages, f)


def manage_chat_history():
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]

def qna_chatbot():
    selected_language = "en"
    with st.container():
        col1, col2, col3 = st.columns([0.6, 0.3, 0.1], vertical_alignment="bottom")
        with col1:
            st.title("Legal Q&A Chatbot")
        with col2:
            selected_language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], label_visibility="hidden")
        with col3:
            if st.button("Back", help="Go to Home"):
                st.session_state.page = "home"
                st.rerun()

    messages = load_qna_history()

    def get_legal_response(user_input, language):
        system_prompt = (
            "You are VidhiVani, a legal assistant specializing strictly in Indian law. "
            "You must only provide factual and publicly available legal information based on Indian statutes, case law, and established legal principles. "
            "The user interacting with you has absolutely no legal knowledge, so construct your responses in a very simple yet informative manner. "
            "You must adhere to the following strict guidelines: "
            "\n\n1. Do NOT answer any question that is not directly related to Indian law. If a user asks a non-legal question, do not respond to it but rather politely apologize that you cannot provide an answer to that. "
            "Do NOT provide opinions, or strategic guidance. "
            "Do NOT discuss legal matters outside the jurisdiction of India. "
            "\n\nYour responses must be: "
            "\n- Strictly factual, objective, and legally accurate. "
            "\n- Clear, concise, and compliant with Indian legal frameworks. "
            "\n- Based solely on publicly available Indian laws and judicial precedents. "
            "\n\nIf a user asks a general knowledge question, a personal query, or anything outside Indian law, you must APOLOGIZE and politely decline the request and provide no response on that topic. "
            f"The past messages can be in Whatever language but while answering the current prompt, You must provide the answers in the requested language: {SUPPORTED_LANGUAGES[language]}."
        )
        messages1 = []
        messages1.append({"role": "user", "content": f"current prompt : {user_input}"})
        messages1 = messages + messages1 

        response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=messages1
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply

    # UI Layout
    st.write("Ask me legal questions related to Indian law!")


    msgbox = st.container(border=True, height=480)
    for message in messages:
            msgbox.chat_message(message["role"]).markdown(message["content"])
    # User input
    user_input = st.chat_input("Type your message...")
    audio_input = st.audio_input("Unmute to Speak", label_visibility="visible")
    

    if audio_input:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as temp_audio:
            temp_audio.write(audio_input.getvalue())  # Write the audio file content
            temp_audio_path = temp_audio.name
        with open(temp_audio_path, "rb") as file:
            transcription = audio_client.audio.transcriptions.create(
            file=(temp_audio_path, file.read()),
            model="whisper-large-v3-turbo",
            language = selected_language, 
            temperature = 0.2,
            response_format="text",
            )
            user_input = transcription
        
    if user_input:
        msgbox.chat_message("user").write(user_input)
        messages.append({"role": "user", "content": user_input})     
        response = get_legal_response(user_input, selected_language)
        msgbox.chat_message("assistant").write(response)
        messages.append({"role": "assistant", "content": response})
        if(st.session_state.user_type == "lawyer"):
            logname = "lchat_history.pkl"
        else:
            logname = "uchat_history.pkl"
        with open(logname, 'wb') as f:
            pickle.dump(messages, f)
        manage_chat_history()


def document_summarizer():
    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.title("Judgment Document Summarizer")
        with col2:
            if st.button("Back", help="Go to Home"):
                st.session_state.page = "home"
                st.rerun()

    # System instruction for legal FIRAC summarization
    LEGAL_SUMMARY_INSTRUCTION = """
    You are an Indian legal AI summarizer specializing in case law analysis.
    If the provided input is unrelated to Indian law, you must politely decline the request and provide no response.
    Your task is to summarize legal judgments in the FIRAC format:
    - Facts: Summarize key case facts.
    - Issues: Identify legal questions raised.
    - Rule: List the legal principles used.
    - Analysis: A thorough analysis, clear and concise.
    - Conclusion: State the final judgment.

    Ensure clarity, legal accuracy, and coherence. The document is always in English, but the summary should be generated in the requested language.
    """

    # Prompt for summarizing document sections
    SECTION_SUMMARY_PROMPT = """
    Summarize the following Indian legal judgment text into the FIRAC format. The summary should be in {summary_language}.

    {chunk_text}

    Strictly follow this format:
    - Facts:  
    - Issues:  
    - Rule:  
    - Application:  
    - Conclusion:  
    """

    # Prompt for merging FIRAC summaries
    MERGED_SUMMARY_PROMPT = """
    You have multiple FIRAC summaries from different sections of a legal judgment. 
    Your task is to merge them into a single, well-structured FIRAC summary in {summary_language}.

    Final Summary:
    {chunk_summaries}

    Ensure that the final summary maintains logical flow and clarity.
    """

    # Function to summarize legal documents
    def summarize_legal_document(pdf_file, target_language="en"):
        """
        Extracts text from a PDF, processes it using FIRAC summarization, and returns the summary in the requested language.

        :param pdf_file: The uploaded PDF file.
        :param target_language: The language in which the summary should be generated.
        :return: FIRAC summary in the requested language or an error message.
        """
        if target_language not in SUPPORTED_LANGUAGES:
            return f"Error: The selected language is not supported. Please choose from: {', '.join(SUPPORTED_LANGUAGES.values())}."

        try:
            # Save the uploaded file as a temporary PDF
            with tempfile.NamedTemporaryFile(mode='wb+', delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_file.getvalue())
                temp_pdf.flush()
                temp_file_path = temp_pdf.name

            # Load and extract text from the PDF
            pdf_loader = PyPDFLoader(temp_file_path)
            extracted_docs = pdf_loader.load()
            full_text = "\n".join([doc.page_content for doc in extracted_docs])

            # Improved text splitting using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            text_chunks = text_splitter.split_text(full_text)

            # Summarize each section separately
            section_summaries = []
            for chunk in text_chunks:
                formatted_prompt = SECTION_SUMMARY_PROMPT.format(
                    chunk_text=chunk, 
                    summary_language=SUPPORTED_LANGUAGES[target_language]
                )
                response = client.chat.completions.create(
                    model="databricks-meta-llama-3-3-70b-instruct",
                    messages=[
                        {"role": "system", "content": LEGAL_SUMMARY_INSTRUCTION},
                        {"role": "user", "content": formatted_prompt}
                    ]
                )
                section_summaries.append(response.choices[0].message.content) 

            # Merge summarized sections into a single summary
            merge_prompt = MERGED_SUMMARY_PROMPT.format(
                chunk_summaries="\n\n".join(section_summaries),
                summary_language=SUPPORTED_LANGUAGES[target_language]
            )
            final_response = client.chat.completions.create(
                model="databricks-meta-llama-3-3-70b-instruct",
                messages=[
                    {"role": "system", "content": LEGAL_SUMMARY_INSTRUCTION},
                    {"role": "user", "content": merge_prompt}
                ]
            )
            return final_response.choices[0].message.content
        
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    # Streamlit UI Components
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    selected_language = st.selectbox("Select summary language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x])
    
    if st.button("Summarize Document"):
        if uploaded_file is not None:
            summary_result = summarize_legal_document(uploaded_file, selected_language)
            st.subheader("Generated FIRAC Summary")
            st.write(summary_result)
        else:
            st.warning("Please upload a PDF file before summarizing.")

def contract_analyser():
    selected_language = "en"
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "kn": "Kannada",
        "mr": "Marathi",
        "ml": "Malayalam",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "or": "Odia",
        "as": "Assamese"
    }

    def extract_text_from_pdf(file_content):
        """Extracts text from an uploaded PDF file."""
        text = ""
        file_like_object = BytesIO(file_content.getvalue())
        reader = PyPDF2.PdfReader(file_like_object)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    def classify_document(document_text):
        """Uses an LLM to determine if the document is a contract."""
        classification_prompt = (
            "You are a legal AI trained to classify documents. "
            "Determine if the given text is from a *legally binding contract*. "
            "Respond with only 'Yes' or 'No'."
        )

        response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": document_text[:2000]}  # Limit input for classification
            ]
        )

        return response.choices[0].message.content.strip().lower() == "yes"

    def summarize_contract(contract_text, language="en"):
        """Summarizes the contract and provides risk analysis using an LLM."""

        if language not in SUPPORTED_LANGUAGES:
            return f"⚠ Language '{language}' is not supported. Choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}"

        system_prompt = (
            "You are VidhiVani, a legal assistant specializing in contract analysis. "
            "You must *STRICTLY* summarize only legal contracts and provide risk analysis. "
            "Extract and summarize key legal aspects: involved parties, obligations, penalties, liabilities, "
            "termination clauses, jurisdiction, and dispute resolution methods. "
            "If a non-contract document is provided, politely refuse analysis. "
            f"Provide responses in: {SUPPORTED_LANGUAGES[language]}."
        )

        response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contract_text}
            ]
        )

        assistant_reply = response.choices[0].message.content
        with st.container():
            st.markdown(assistant_reply)

        return assistant_reply

    with st.container():
        selected_language = "en"
        col1, col2, col3 = st.columns([0.6, 0.3, 0.1], vertical_alignment="bottom")
        with col1:
            st.title("Contract Analyzer")
        with col2:
            selected_language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], label_visibility="hidden")
        with col3:
            if st.button("Back", help="Go to Home"):
                st.session_state.page = "home"
                st.rerun()
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if st.button("Analyze Implications") and uploaded_file:
        document_text = extract_text_from_pdf(uploaded_file)
        # Classify document using LLM
        if classify_document(document_text):
            print("✅ This document is a legal contract.")
            summary = summarize_contract(document_text, language=selected_language)  # Change language if needed
            print("\n📄 *Contract Summary & Risk Analysis:*\n", summary)
        else:
            print("❌ This document is NOT a legal contract. Please upload a valid contract.")

    # Store conversation history (Limited to Last 10 messages)
    if "messages" not in st.session_state:
        st.session_state.messages = []

if "user_type" not in st.session_state:
    st.session_state.user_type = None
def main():
    navDic = {"Ask Queries" : "qna", "Summarize Documents": "summarizer", "Analyse Contracts": "contract", "Retrieve prior cases": "PriorCaseRetrieval"}
    navList = ["Ask Queries", "Summarize Documents", "Analyse Contracts", "Retrieve prior cases"] if (st.session_state.user_type == "lawyer")  else ["Ask Queries", "Analyse Contracts"]
    
    st.session_state.page = navDic[st.sidebar.radio(
        'Go to',
        navList
    )]
    def logout_callback():
        st.session_state.clear()
        st.session_state.trigger_rerun = True

    st.sidebar.button("Logout", on_click=logout_callback)

    if st.session_state.get("trigger_rerun", False):
        st.session_state.trigger_rerun = False
        st.rerun()
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "qna":
        qna_chatbot()
    elif st.session_state.page == "summarizer":
        document_summarizer()
    elif st.session_state.page == "contract":
        contract_analyser()
    elif st.session_state.page == "PriorCaseRetrieval":
        rag()
    else:
        home()

if st.session_state.user_type is None:
    st.title("Welcome to VidhiVani")
    st.write("This app provides informational insights and is not a substitute for legal advice.")
    st.write("WHO ARE YOU?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("LAW PRACTITIONER"):
            st.session_state.user_type = "lawyer"
            st.rerun()  # Rerun the script to update the page
    
    with col2:
        if st.button("COMMON PUBLIC"):
            st.session_state.user_type = "client"
            st.rerun()  # Rerun the script to update the page

else:
    main()  # Run the main application after user selection
