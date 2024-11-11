# Chatbot with PDF Querying, Appointment Booking, and Call Back Feature

This project implements a chatbot using LangChain and Google Gemini Pro, enabling users to interact with PDF documents by asking questions. It also includes a feature to request a call back and book appointments.

Features:
1. **PDF Querying:** Allows users to upload PDF files and ask questions based on the content of the document.
2. 	**Call me  Request: **Users can request a call back by providing their name, phone number, and email.
3. 	**Appointment Booking:** Users can schedule an appointment by selecting a date
4. 	**Date Recognition:** Recognizes and validates appointment dates in the YYYY-MM-DD format

Requirements: 
. Python 3.11
. Streamlit
. Google Gemini Pro API Key
. LangChain

Install the required dependencies: pip install -r requirements.txt

Create a .env file and add your Google Gemini Pro API key

Run the Streamlit app:   streamlit run app.py

**Usage:**
1. Upload PDF: Upload your PDF documents using the file uploader.
2. Ask Questions: Type questions based on the content of the uploaded PDF. The bot will respond with relevant information.
3. Call Back Request: Type “call me” to be prompted to enter your name, phone number, and email.
4. Book Appointment: Use the “Book Appointment” button to select a date and schedule an appointment.

	
