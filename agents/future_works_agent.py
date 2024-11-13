from config import model
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from agents import  SearchAgent
import streamlit as st
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "gemini-key"
            
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class FutureWorksAgent:
    def __init__(self):
        self.model = model
        self.prompt = """Generate ideas for future research to be included in a review paper. Provide a well-structured summary that highlights opportunities for future work and potential improvements.

        Chat history:
        {chat_history}

        Research context:
        {context}

        Guidelines:
        1. Determine if the query seeks a combined review of the papers or specific future directions.
        2. If query is for future directions:
            a. Summarize key opportunities for future research and the papers limitations retrived from the papers.
            b. Generate an improvement plan based on influential research works, including suggestions for novel contributions.
        3. If query is for review of paper:
            a. Give a comprehensive and combined review for these papers including sections like Introduction, Methods discussed in papers, Limitations or gaps in research, Future work and conclusion.
            b. Combine insights from multiple papers to create a cohesive roadmap, with clear, actionable directions for new research.
        4. Discuss technical challenges and propose methodological enhancements where applicable.
        5. Highlight potential applications and how future work could address existing gaps.

        Focus on generating a structured, cohesive set of recommendations for advancing the field.
"""
        

    def solve(self, query):
        # Check if search has been performed
        if not os.path.exists("vdb_chunks"):
            st.warning("No papers loaded.")
            
        # Load vector store
        vdb_chunks = FAISS.load_local("vdb_chunks", embeddings, index_name="base_and_adjacent", allow_dangerous_deserialization=True)
        
        # Get chat history
        chat_history = st.session_state.get("chat_history", [])
        chat_history_text = "\n".join([f"{sender}: {msg}" for sender, msg in chat_history[-5:]])
        
        # Get relevant chunks
        retrieved = vdb_chunks.as_retriever().get_relevant_documents(query)
        context = "\n".join([f"{doc.page_content} Source: {doc.metadata['source']}" for doc in retrieved])

    
        
        # Generate response
        full_prompt = self.prompt.format(
            chat_history=chat_history_text,
            context=context
        )
        response = self.model.generate_content(full_prompt)
        return response.text