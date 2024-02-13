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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question, chain, session_state):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Get the last answer from the conversation history
    last_answer = session_state["history"][-1][1] if session_state["history"] else ""
    
    response = chain({"input_documents": docs, "question": user_question, "last_answer": last_answer}, return_only_outputs=True)
    session_state["history"].append((user_question, response["output_text"]))
    return response["output_text"]





# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
#     if pdf_docs and st.button("Submit & Process"):
#         with st.spinner("Processing..."):
#             raw_text = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#         st.success("Done")
        
#         if st.button("Start Chat"):
#             st.experimental_set_query_params(chat=True)

#     if st.query_params.get("chat"):
#         st.title("Chat")
#         user_question = st.text_input("Ask a Question from the PDF Files")
#         if user_question:
#             user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         st.write("Upload your PDF Files and click on the Submit & Process Button to start chatting.")
#         st.write("Once the PDF is processed, click on 'Start Chat' to begin chatting.")

# if __name__ == "__main__":
#     main()




def main():
    st.set_page_config("PDF Chatbot")
    st.title("Chat with PDF using Gemini💬")

    session_state = st.session_state.setdefault("session_state", {"history": []})

    # Upload PDF files
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    # Process PDF files and create vector store
    if pdf_docs and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        st.success("PDF Processing Complete")

    # Create conversational chain
    chain = get_conversational_chain()

    # User input field
    user_question = st.text_input("Ask a question")

    # Respond to user input
    if st.button("Ask"):
        if pdf_docs:
            if "faiss_index" not in os.listdir():
                st.error("Please process the PDF files first.")
            else:
                if user_question:
                    with st.spinner("Thinking..."):
                        response = user_input(user_question, chain, session_state)
                    st.write("Reply:", response)
                else:
                    st.warning("Please ask a question.")
        else:
            st.error("Please upload PDF files.")

    # Display conversation history
    st.title("Conversation History")
    for i, (question, answer) in enumerate(session_state["history"], 1):
        st.write(f"Question {i}: {question}")
        st.write(f"Answer {i}: {answer}")

if __name__ == "__main__":
    main()










# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()