import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save the vector store
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

# process user input and provide response from the document
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Function to handle the appointment form and validation
def call_me():
    name = st.text_input("Enter your Name", key="name_input")
    phone = st.text_input("Enter your Phone Number", key="phone_input")
    email = st.text_input("Enter your Email", key="email_input")

    submit_button = st.button("Schedule Call", key="submit_button")

    if submit_button:
        if not name or not phone or not email:
            st.warning("Please provide your details. All fields are required!")
        else:
            st.success(f"Thank you, {name}. We will call you at {phone}")

# Function to handle the appointment scheduling form and validation
def book_appointment():
    appointment_date = st.date_input("Enter your Appointment Date (YYYY-MM-DD)", key="appointment_date_input")
    submit_button = st.button("Confirm Appointment", key="submit_appointment_button")

    if submit_button:
        if not appointment_date:
            st.warning("Please provide an appointment date!")
        else:
            st.success(f"Congrats, appointment booked for {appointment_date}.")

# Handle the Streamlit app
def main():
    st.set_page_config(page_title="Chat with AI Bot")
    st.header("Chat with AI Bot")

    # File uploader for PDF documents
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if pdf_docs:
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Document processed successfully!")

    # User input for question
    user_question = st.text_input("Ask a question from the PDF OR type 'call me' to schedule a call")

    if user_question:
        if "call me" in user_question.lower():
            call_me()
        else:
            # Else, handle the document-based question
            user_input(user_question)

    # "Book Appointment" button
    st.header("Book Your Appointment")
    book_appointment()

if __name__ == "__main__":
    main()