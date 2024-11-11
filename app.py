import streamlit as st
import re
from datetime import datetime
import dateparser
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Generate QA chain for chatbot using Gemini Pro
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
    provided in context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Validate email using regex
def validate_email(email):
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(email_regex, email)

# Validate phone number using regex
def validate_phone(phone):
    phone_regex = r"^\+?[1-9]\d{1,14}$"
    return re.match(phone_regex, phone)

# Process user input and provide response from the document
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Handle 'call me' conversation flow
def call_me():
    if 'call_me_step' not in st.session_state:
        st.session_state.call_me_step = 0
        st.session_state.name = ""
        st.session_state.phone = ""
        st.session_state.email = ""

    #  Ask for name
    if st.session_state.call_me_step == 0:
        st.write("Hello! What's your name?")
        st.session_state.name = st.text_input("Your name", key="call_name_input")
        if st.session_state.name:
            st.session_state.call_me_step = 1

    #  Ask for phone number
    elif st.session_state.call_me_step == 1:
        st.write(f"Nice to meet you, {st.session_state.name}! What is your phone number?")
        st.session_state.phone = st.text_input("Phone number", key="call_phone_input")
        if st.session_state.phone and validate_phone(st.session_state.phone):
            st.session_state.call_me_step = 2
        elif st.session_state.phone:
            st.warning("Please enter a valid phone number.")

    #  Ask for email
    elif st.session_state.call_me_step == 2:
        st.write(f"Thanks! Now, please provide your email address.")
        st.session_state.email = st.text_input("Email", key="call_email_input")
        if st.session_state.email and validate_email(st.session_state.email):
            st.session_state.call_me_step = 3
        elif st.session_state.email:
            st.warning("Please enter a valid email address.")

    #  Confirmation
    elif st.session_state.call_me_step == 3:
        st.write(f"Thank you, {st.session_state.name}! We will call you at {st.session_state.phone} soon.")
        if st.button("Confirm Call"):
            st.success("Your call request has been received. We will contact you shortly.")

# Handle 'book appointment' conversation flow
def book_appointment():
    if 'book_appointment_step' not in st.session_state:
        st.session_state.book_appointment_step = 0
        st.session_state.name = ""
        st.session_state.phone = ""
        st.session_state.appointment_date = ""

    # Ask for name
    if st.session_state.book_appointment_step == 0:
        st.write("Hello! What's your name?")
        st.session_state.name = st.text_input("Your name", key="book_name_input")
        if st.session_state.name:
            st.session_state.book_appointment_step = 1

    #  Ask for phone number
    elif st.session_state.book_appointment_step == 1:
        st.write(f"Nice to meet you, {st.session_state.name}! What is your phone number?")
        st.session_state.phone = st.text_input("Phone number", key="book_phone_input")
        if st.session_state.phone and validate_phone(st.session_state.phone):
            st.session_state.book_appointment_step = 2
        elif st.session_state.phone:
            st.warning("Please enter a valid phone number.")

    # Ask for appointment date
    elif st.session_state.book_appointment_step == 2:
        st.write("Great! When would you like to book the appointment? (e.g., 'Next Monday')")
        appointment_input = st.text_input("Appointment Date", key="book_date_input")
        if appointment_input:
            parsed_date = dateparser.parse(appointment_input)
            if parsed_date:
                st.session_state.appointment_date = parsed_date.date()
                st.session_state.book_appointment_step = 3
            else:
                st.warning("Please enter a valid date.")

    # Confirmation
    elif st.session_state.book_appointment_step == 3:
        st.write(f"Thank you, {st.session_state.name}! Your appointment is booked for {st.session_state.appointment_date}.")
        if st.button("Confirm Appointment"):
            st.success("Appointment successfully booked!")

# Streamlit app interface
def main():
    st.set_page_config(page_title="Chatbot")
    st.header("Chat with AI Bot")

    # PDF Upload and Processing
    st.subheader("Upload a PDF document")
    pdf_docs = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)

    if pdf_docs:
        pdf_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(pdf_text)
        get_vector_store(text_chunks)

        st.write("PDF processed successfully! You can now ask questions related to the document.")
        user_question = st.text_input("Ask a question about the document")

        if user_question:
            user_input(user_question)

    # Display conversation flow (chatbot)
    user_question = st.text_input("Type 'call me' to Schedule a Call or 'book appointment' to book an appointment")

    if user_question:
        if "call me" in user_question.lower():
            call_me()
        elif "book appointment" in user_question.lower():
            book_appointment()

if __name__ == "__main__":
    main()
