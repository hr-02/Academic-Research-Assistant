
from config import model
import urllib.request as libreq
import xml.etree.ElementTree as ET
import requests
import os
from langchain.document_loaders import PDFMinerLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

os.makedirs("papers", exist_ok=True)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "gemini-key"
            
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

text_splitter   =   RecursiveCharacterTextSplitter(
            chunk_size=700, # Specify the character chunk size
            chunk_overlap=50, # "Allowed" Overlap across chunks
            length_function=len # Function used to evaluate the chunk size (here in terms of characters)
        )





class SearchAgent:
    def __init__(self):
     
        self.model = model
        self.p = """You are an assistant designed to extract research topics or titles from user queries. When a user asks about a specific topic, identify the central subject of their query and provide a concise, clear title or topic related to that area of research. If the query refers to a particular research paper, include the paper's title, author(s), and publication year.

            Here are the instructions you should follow:

            General Topics: If the query mentions a general topic without referring to a specific paper, identify the primary research area or topic. For example, if the query is "What are the advancements in text-to-SQL models?" your response should be simply "Text to SQL Models."

            Specific Research Papers: If the query mentions a particular paper, extract the title, author(s), and year of the paper. For example, if the query is "What did the paper by Edie Edwards in 2023 say about AI in Software Development?" your response should be "AI in Software Development (Edie Edwards, 2023)."

            Abstract or General Query: If the query is an abstract or general inquiry into a topic, return the main theme or title of that topic. For instance, "What are the advancements in natural language processing?" would result in "Advancements in Natural Language Processing."

            Examples:

            User Query: "Tell me about recent advancements in LLM Safety." Response: "LLM Safety."

            User Query: "What does the paper 'Deep Learning for Trash Detection by Gupta, 2024' cover?" Response: "Deep Learning for Trash Detection (Gupta, 2024)." """
    
    def solve(self, task):
        print(f"Searching for information on: {task}")
        response = model.generate_content(self.p+task)
        query =  response.text.strip()


        r=query.split(" ")
        query="%20".join(r)
        
        with libreq.urlopen(f'''http://export.arxiv.org/api/query?search_query=all:{query}&sortBy=relevance&sortOrder=descending&start=0&max_results=5''') as url:
            r = url.read()
            
            

        xml_content = r
        root = ET.fromstring(xml_content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        ids = [entry.find('atom:id', ns).text for entry in root.findall('atom:entry', ns)]
        pdf_urls = [url.replace("abs", "pdf") for url in ids]

        # Create list to store paper information
        papers = []
        
        # Extract information for each paper
        for entry in root.findall('atom:entry', ns):
            paper_info = {}
            
            # Get paper title
            title = entry.find('atom:title', ns).text
            paper_info['title'] = title
            
            # Get paper ID and create PDF link
            paper_id = entry.find('atom:id', ns).text
            pdf_link = paper_id.replace("abs", "pdf")
            paper_info['link'] = pdf_link
            
            # Get publication year from published date
            published = entry.find('atom:published', ns).text
            year = published[:4]  # Extract year from date string
            paper_info['year'] = year
            
            papers.append(paper_info)

        adjacents_papers_numbers = []

        def download_pdf_paper_from_url(url):
            paper_number = os.path.basename(url).strip(".pdf")
            res = requests.get(url)
            pdf_path = f"papers/{paper_number}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(res.content)
            return paper_number
            
        for paper in papers:
            paper_number = download_pdf_paper_from_url(paper['link'])
            adjacents_papers_numbers.append(paper_number)
            # Add paper number to paper info
            paper['paper_number'] = paper_number

        vdb_chunks = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        for pdf_number in adjacents_papers_numbers:
            docs = ArxivLoader(query=pdf_number)
            docs = PDFMinerLoader(f"papers/{pdf_number}.pdf").load()
            docs = text_splitter.split_documents(docs)
            vdb_chunks.add_documents(docs)
        
        vdb_chunks.save_local("vdb_chunks", index_name="base_and_adjacent")
        
        return papers , papers # Return the list of paper dictionaries
        
     

