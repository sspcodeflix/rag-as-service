import streamlit as st
import requests
from anthropic import Anthropic
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse

class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str, serpapi_api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline with API keys.
        """
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.serpapi_api_key = serpapi_api_key
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        
        # API endpoints for document service
        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"
    
    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        """
        Upload a document to Ragie from a URL.
        """
        if not name:
            name = urlparse(url).path.split('/')[-1] or "document"
            
        payload = {
            "mode": mode,
            "name": name,
            "url": url
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }
        
        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers, timeout=10)
        
        if not response.ok:
            raise Exception(f"Document upload failed: {response.status_code} {response.reason}")
            
        return response.json()
    
    def retrieve_chunks(self, query: str, scope: str = "tutorial") -> List[str]:
        """
        Retrieve relevant chunks from Ragie for a given query.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
        }
        
        payload = {
            "query": query,
            "filters": {
                "scope": scope
            }
        }
        
        response = requests.post(
            self.RAGIE_RETRIEVAL_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if not response.ok:
            raise Exception(f"Retrieval failed: {response.status_code} {response.reason}")
            
        data = response.json()
        return [chunk["text"] for chunk in data.get("scored_chunks", [])]
    
    def retrieve_web_results(self, query: str, num_results: int = 3) -> List[str]:
        """
        Retrieve web search results from SerpApi based on the query.
        """
        if not self.serpapi_api_key:
            return []
        search_url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": num_results
        }
        response = requests.get(search_url, params=params, timeout=10)
        if not response.ok:
            raise Exception(f"Web search failed: {response.status_code} {response.reason}")
        data = response.json()
        results = []
        if "organic_results" in data:
            for result in data["organic_results"][:num_results]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                results.append(f"**{title}**: {snippet}")
        return results

    def create_system_prompt(self, doc_chunks: List[str], web_results: List[str]) -> str:
        """
        Create the system prompt with the retrieved document chunks and web search results.
        """
        prompt_parts = []
        if doc_chunks:
            joined_doc_chunks = "\n\n".join(doc_chunks)
            prompt_parts.append(f"Document Information:\n===\n{joined_doc_chunks}\n===")
        if web_results:
            joined_web_results = "\n\n".join(web_results)
            prompt_parts.append(f"Web Search Results:\n===\n{joined_web_results}\n===")
        
        context_info = "\n\n".join(prompt_parts)
        
        return f"""These are very important instructions: You are "Ragie AI", a professional but friendly AI chatbot assisting the user. Your task is to answer the user based on the information provided below. Answer informally, directly, and concisely, including all relevant details. Use Markdown for formatting (e.g., **bold**, *italic*, lists, etc.) and $$ for LaTeX where appropriate. Organize your answer into sections if needed. Do not include raw IDs or sensitive information.

Here is the context available for you:
{context_info}

If the context is missing or insufficient, please indicate that the available information might be incomplete. END SYSTEM INSTRUCTIONS"""

    def generate_response(self, system_prompt: str, query: str) -> str:
        """
        Generate a response using Anthropic's language model.
        """
        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return message.content[0].text if message.content and isinstance(message.content, list) else message.content

    def process_query(self, query: str, scope: str = "tutorial") -> str:
        """
        Process a query through the complete RAG pipeline, using both document chunks and web search results.
        """
        doc_chunks = self.retrieve_chunks(query, scope)
        web_results = self.retrieve_web_results(query) if self.serpapi_api_key else []
        
        if not doc_chunks and not web_results:
            return "No relevant information found for your query."
        
        system_prompt = self.create_system_prompt(doc_chunks, web_results)
        return self.generate_response(system_prompt, query)

def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'api_keys_submitted' not in st.session_state:
        st.session_state.api_keys_submitted = False

def reset_state():
    """Reset session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

def main():
    st.set_page_config(page_title="RAG-as-a-Service", layout="wide")
    initialize_session_state()
    
    # Reset button at the top of the main page
    if st.button("Reset All"):
        reset_state()
    
    st.title("🔗 RAG-as-a-Service")
    st.markdown(
        "This application allows you to upload a document and then query it using a Retrieval Augmented Generation (RAG) pipeline powered by Ragie, Anthropic's Claude, and supplemented by web search results from SerpApi."
    )
    
    # Use tabs to organize the UI
    tabs = st.tabs(["API Keys", "Document Upload", "Query Document"])
    
    # --- API Keys Tab ---
    with tabs[0]:
        st.subheader("🔑 API Keys Configuration")
        with st.form(key="api_keys_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ragie_key = st.text_input(
                    "Ragie API Key", 
                    type="password", 
                    key="ragie_key_input", 
                    help="Obtain your Ragie API key from https://ragie.ai"
                )
            with col2:
                anthropic_key = st.text_input(
                    "Anthropic API Key", 
                    type="password", 
                    key="anthropic_key_input", 
                    help="Obtain your Anthropic API key from https://anthropic.com"
                )
            with col3:
                serpapi_key = st.text_input(
                    "SerpApi API Key", 
                    type="password", 
                    key="serpapi_key_input", 
                    help="Obtain your SerpApi API key from https://serpapi.com"
                )
            submit_api = st.form_submit_button("Submit API Keys")
            
            if submit_api:
                if ragie_key and anthropic_key:
                    try:
                        st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key, serpapi_key)
                        st.session_state.api_keys_submitted = True
                        st.success("API keys configured successfully!")
                    except Exception as e:
                        st.error(f"Error configuring API keys: {str(e)}")
                else:
                    st.error("Please provide at least the Ragie and Anthropic API keys.")
    
    # --- Document Upload Tab ---
    with tabs[1]:
        st.subheader("📄 Document Upload")
        if not st.session_state.api_keys_submitted:
            st.info("Please configure the API keys first (see the API Keys tab).")
        else:
            with st.form(key="upload_form"):
                doc_url = st.text_input("Enter Document URL", key="doc_url")
                doc_name = st.text_input("Document Name (optional)", key="doc_name")
                upload_mode = st.selectbox("Upload Mode", ["fast", "accurate"], key="upload_mode")
                submit_upload = st.form_submit_button("Upload Document")
                
                if submit_upload:
                    if doc_url:
                        try:
                            with st.spinner("Uploading document..."):
                                st.session_state.pipeline.upload_document(
                                    url=doc_url,
                                    name=doc_name if doc_name else None,
                                    mode=upload_mode
                                )
                                # Replace time.sleep with a proper polling mechanism if possible.
                                time.sleep(5)  # Simulate waiting for indexing.
                                st.session_state.document_uploaded = True
                                st.success("Document uploaded and indexed successfully!")
                        except Exception as e:
                            st.error(f"Error uploading document: {str(e)}")
                    else:
                        st.error("Please provide a document URL.")
    
    # --- Query Document Tab ---
    with tabs[2]:
        st.subheader("🔍 Query Document")
        if not st.session_state.document_uploaded:
            st.info("Please upload a document first (see the Document Upload tab).")
        else:
            with st.form(key="query_form"):
                query = st.text_area("Enter your query", key="query_input", height=150)
                submit_query = st.form_submit_button("Generate Response")
                
                if submit_query:
                    if query:
                        try:
                            with st.spinner("Generating response..."):
                                response = st.session_state.pipeline.process_query(query)
                                st.markdown("### Response:")
                                st.markdown(response)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                    else:
                        st.error("Please enter a query.")
    
    # Developer credit at the bottom of the page
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Developed with ❤️ by Soumya Sourav</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
