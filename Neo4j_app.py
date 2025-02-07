import os
import logging
import asyncio
import uuid  # <-- for generating truly unique doc_ids
from datetime import datetime
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF for PDF
import gradio as gr
from docx import Document as DocxDocument
from dotenv import load_dotenv

# Neo4j & LangChain imports
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from neo4j import GraphDatabase

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class Config:
    def __init__(self):
        load_dotenv()
        
        # Neo4j
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', '')
        
        # Provider configuration
        self.provider_type = os.getenv('PROVIDER_TYPE', 'openai')  # 'openai' or 'azure'
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.openai_embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        
        self._validate()

    def _validate(self):
        # Basic checks
        if not self.neo4j_uri or not self.neo4j_username:
            raise ValueError("Missing Neo4j connection details")
        if self.provider_type == 'openai' and not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI usage")

# -------------------------------------------------------------------------
# Neo4j Connection / Helper
# -------------------------------------------------------------------------
class Neo4jHelper:
    """A small helper class to handle Neo4j connections & queries."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query: str, parameters: Optional[Dict] = None):
        with self.driver.session() as session:
            return session.run(query, parameters or {}).data()

# -------------------------------------------------------------------------
# GraphRAG Processor
# -------------------------------------------------------------------------
class GraphRAGProcessor:
    def __init__(self):
        self.config = Config()
        
        # Set environment variables for OpenAI
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # Create or open Neo4j connection
        self.neo4j = Neo4jHelper(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_username,
            password=self.config.neo4j_password
        )
        
        # Create LLM & embeddings
        self.chat_model = ChatOpenAI(
            model_name=self.config.openai_model,
            temperature=0.2
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.config.openai_embedding_model
        )
        
        self._setup_db()

    def _setup_db(self):
        """Ensure constraints or indexes exist in Neo4j, if needed."""
        try:
            # For unique IDs on Document and Chunk nodes
            constraint_queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            ]
            for cq in constraint_queries:
                self.neo4j.run_query(cq)
            
            # If your Neo4j supports vector indexes, you can create them here.
        except Exception as e:
            logging.error(f"Error setting up Neo4j constraints / indexes: {e}")

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF, DOCX, or TXT files."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            text = []
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text.append(page.get_text())
            return "\n".join(text)
        
        elif ext == '.docx':
            doc = DocxDocument(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")

    async def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Dict:
        """
        Process a document:
        1. Extract text
        2. Split into chunks
        3. Embed each chunk
        4. Store in Neo4j
        """
        try:
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                return {"status": "error", "message": "Empty file or text extraction failed."}

            # Split text into smaller segments
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            document_name = os.path.basename(file_path)
            
            # Use a truly unique doc_id for each new upload
            unique_id = uuid.uuid4().hex
            doc_id = f"doc-{document_name}-{unique_id}"

            # Create Document node
            self.neo4j.run_query(
                """
                CREATE (d:Document {id: $doc_id})
                SET d.file_name = $file_name, d.upload_date = timestamp()
                """,
                {"doc_id": doc_id, "file_name": document_name}
            )
            
            # Insert each chunk into Neo4j
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}-chunk-{idx}"
                
                # Embed chunk
                embedding = self.embeddings.embed_query(chunk_text)
                
                self.neo4j.run_query(
                    """
                    CREATE (c:Chunk {id: $chunk_id})
                    SET c.text = $chunk_text, 
                        c.embedding = $embedding,
                        c.index = $idx
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "embedding": embedding,
                        "idx": idx,
                        "doc_id": doc_id
                    }
                )
            return {
                "status": "success",
                "file": document_name,
                "chunks_stored": len(chunks)
            }
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return {"status": "error", "message": str(e)}

    def similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find the most relevant chunks via embeddings & manual cosine similarity."""
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve all chunks
        all_chunks = self.neo4j.run_query("""
            MATCH (c:Chunk)
            RETURN c.id AS chunk_id, c.text AS chunk_text, c.embedding AS embedding
        """)
        
        def dot_product(a, b):
            return sum(x*y for x, y in zip(a, b))
        
        def norm(a):
            return sum(x*x for x in a)**0.5
        
        results = []
        for item in all_chunks:
            emb = item['embedding']
            similarity = dot_product(query_embedding, emb) / (norm(query_embedding) * norm(emb))
            results.append({
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "score": similarity
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def query_knowledge(self, user_query: str) -> Dict:
        """Perform a simple RAG query: get top chunks & create an answer via ChatOpenAI."""
        try:
            top_chunks = self.similarity_search(user_query, limit=5)
            
            if not top_chunks:
                return {
                    "answer": "No relevant information found in the database.",
                    "search_results": []
                }
            
            # Create prompt
            context_text = "\n\n".join(
                f"Chunk (score={round(c['score'],2)}): {c['chunk_text']}" 
                for c in top_chunks
            )
            system_prompt = (
                "You are a helpful assistant that uses the provided context to answer user questions.\n"
                "Use only the context below to answer accurately.\n\n"
                f"Context:\n{context_text}\n\n"
                "Answer the user's question as best you can."
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.chat_model.agenerate([[m for m in messages]])
            answer_text = response.generations[0][0].text.strip()
            
            return {
                "answer": answer_text,
                "search_results": top_chunks
            }
        except Exception as e:
            logging.error(f"Query error: {e}")
            return {"answer": f"Error: {str(e)}", "search_results": []}

# -------------------------------------------------------------------------
# Gradio Chat Interface
# -------------------------------------------------------------------------
class ChatInterface:
    """A simple Gradio interface for the GraphRAGProcessor."""
    
    def __init__(self, processor: GraphRAGProcessor):
        self.processor = processor
        self.interface = self._setup_gradio()

    def _setup_gradio(self) -> gr.Blocks:
        with gr.Blocks(title="GraphRAG", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# GraphRAG: Ask Your Documents in Neo4j")

            with gr.Tab("Chat"):
                chat_history_box = gr.Textbox(
                    label="Conversation History",
                    value="(Conversation will appear here)",
                    lines=15,
                    interactive=False
                )
                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type a question...",
                    lines=2
                )
                send_button = gr.Button("Send")

                current_answer = gr.Textbox(
                    label="Answer",
                    interactive=False
                )
                search_result_df = gr.DataFrame(
                    headers=["Chunk ID", "Similarity Score", "Preview"],
                    label="Search Results",
                    interactive=False
                )

                async def chat_handler(question, chat_history):
                    if not question.strip():
                        return chat_history, "", []

                    history_text = f"{chat_history}\n\nUser: {question}"
                    
                    query_res = await self.processor.query_knowledge(question)
                    answer = query_res.get("answer", "")
                    top_chunks = query_res.get("search_results", [])

                    table_data = [
                        [
                            c["chunk_id"],
                            round(c["score"], 3),
                            c["chunk_text"][:70] + "..." if len(c["chunk_text"]) > 70 else c["chunk_text"]
                        ]
                        for c in top_chunks
                    ]
                    
                    updated_history = history_text + f"\nAssistant: {answer}"
                    return updated_history, answer, table_data

                send_button.click(
                    fn=chat_handler,
                    inputs=[user_input, chat_history_box],
                    outputs=[chat_history_box, current_answer, search_result_df]
                )

            with gr.Tab("Process Document"):
                file_input = gr.File(label="Upload a PDF, DOCX, or TXT file")
                chunk_size_slider = gr.Slider(100, 2000, step=100, value=500, label="Chunk Size")
                overlap_slider = gr.Slider(0, 500, step=50, value=100, label="Chunk Overlap")
                status_box = gr.Textbox(label="Status", interactive=False)
                
                process_button = gr.Button("Process File")

                async def process_handler(file, chunk_size, overlap):
                    if not file:
                        return "No file uploaded."
                    result = await self.processor.process_document(
                        file.name,
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    if result["status"] == "success":
                        return f"Processed '{result['file']}' with {result['chunks_stored']} chunks stored."
                    else:
                        return f"Error: {result['message']}"

                process_button.click(
                    fn=process_handler,
                    inputs=[file_input, chunk_size_slider, overlap_slider],
                    outputs=[status_box]
                )

        return demo

    def launch(self):
        # Set share=True to create a public link
        self.interface.launch(server_port=7860, share=True)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    processor = GraphRAGProcessor()
    chat_app = ChatInterface(processor)
    chat_app.launch()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
