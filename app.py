"""
Streamlit UI for Hybrid RAG System
Graph RAG (Neo4j) + Vector RAG with CrewAI - Fully Agent-Driven
"""

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from typing import List, Dict, Optional
import os
from pathlib import Path
from datetime import datetime
import json
import base64

load_dotenv()


class HybridRAGSystem:
    def _resolve_receipt_path(self, relative_path: str) -> str:
        """Convert relative receipt path to absolute path"""
        if not relative_path:
            return None

        # If already absolute, return as is
        if os.path.isabs(relative_path):
            return relative_path

        # Remove leading ./ if present
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        elif relative_path.startswith(".\\"):
            relative_path = relative_path[2:]

        # Join with app root to get absolute path
        absolute_path = os.path.join(self.app_root, relative_path)

        return absolute_path

    def _query_graph_for_agent(self, cypher_query: str) -> str:
        """Helper method for graph queries"""
        try:
            result = self.graph.query(cypher_query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _search_vector_for_agent(
        self,
        query: str,
        product_id: Optional[str] = None,
        receipt_path: Optional[str] = None,
    ) -> str:
        """Helper method for vector search - can load PDF on-the-fly"""
        try:
            # Resolve receipt path to absolute if provided
            if receipt_path:
                receipt_path = self._resolve_receipt_path(receipt_path)

            # If a specific receipt path is provided, load it on-the-fly
            if receipt_path and os.path.exists(receipt_path):
                # Load and embed this specific PDF temporarily
                loader = PyPDFLoader(receipt_path)
                documents = loader.load()

                for doc in documents:
                    if product_id:
                        doc.metadata["product_id"] = product_id
                    doc.metadata["source"] = receipt_path

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)

                # Create temporary vector store for this specific document
                temp_embeddings = self.embeddings
                temp_vector_store = Chroma.from_documents(
                    documents=splits, embedding=temp_embeddings
                )

                # Search in this specific document
                results = temp_vector_store.similarity_search(query, k=4)

                formatted_results = []
                for i, doc in enumerate(results, 1):
                    formatted_results.append(
                        f"--- Document Chunk {i} ---\n"
                        f"Source: {receipt_path}\n"
                        f"Content:\n{doc.page_content}\n"
                    )

                return (
                    "\n".join(formatted_results)
                    if formatted_results
                    else f"Searched receipt at {receipt_path} but no relevant information found for the query."
                )

            # Otherwise search existing vector store
            if self.vector_store is None:
                return "No receipt data available in vector store."

            filter_dict = {"product_id": product_id} if product_id else None

            results = self.vector_store.similarity_search(
                query, k=4, filter=filter_dict
            )

            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(
                    f"--- Document Chunk {i} ---\n"
                    f"Product ID: {doc.metadata.get('product_id', 'N/A')}\n"
                    f"Content:\n{doc.page_content}\n"
                )

            return (
                "\n".join(formatted_results)
                if formatted_results
                else "No relevant receipts found."
            )
        except Exception as e:
            return f"Error searching vector store: {str(e)}"

    def _get_all_products_for_agent(self) -> str:
        """Helper method to get all products"""
        try:
            cypher = """
            MATCH (p:Product)
            RETURN p.id as id, p.name as name, p.category as category
            ORDER BY p.name
            """
            result = self.graph.query(cypher)
            return json.dumps(result, indent=2) if result else "No products found."
        except Exception as e:
            return f"Error retrieving products: {str(e)}"

    def _get_product_with_receipt_path(self, product_identifier: str) -> Optional[Dict]:
        """Get product details including receipt path if available"""
        try:
            cypher = """
            MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS toLower($identifier) OR p.id = $identifier
            OPTIONAL MATCH (p)-[:HAS_RECEIPT]->(r:Receipt)
            RETURN p.id as id, p.name as name, p.price as price,
                   p.specs as specs, p.manufacturer as manufacturer,
                   p.category as category, r.path as receipt_path
            LIMIT 1
            """
            result = self.graph.query(cypher, {"identifier": product_identifier})
            return result[0] if result else None
        except Exception as e:
            return None

    def _add_to_memory(self, user_query: str, response: str, context: Dict):
        """Add conversation to memory, keeping only last 5"""
        memory_entry = {
            "user_query": user_query,
            "response": response,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        self.conversation_memory.append(memory_entry)

        # Keep only last 5 conversations
        if len(self.conversation_memory) > self.max_memory_length:
            self.conversation_memory = self.conversation_memory[
                -self.max_memory_length :
            ]

    def _get_conversation_context(self) -> str:
        """Format conversation memory for agents"""
        if not self.conversation_memory:
            return "No previous conversation history."

        context_lines = ["Recent Conversation History (Last 5 exchanges):"]
        context_lines.append("=" * 60)

        for i, entry in enumerate(self.conversation_memory, 1):
            context_lines.append(f"\n**Exchange {i}:**")
            context_lines.append(f"User: {entry['user_query']}")
            context_lines.append(
                f"Assistant: {entry['response'][:200]}..."
            )  # Truncate long responses

            # Add important context
            if entry["context"].get("products_mentioned"):
                context_lines.append(
                    f"Products mentioned: {', '.join(entry['context']['products_mentioned'])}"
                )
            if entry["context"].get("receipt_used"):
                context_lines.append(
                    f"Receipt accessed: {entry['context']['receipt_used']}"
                )

        context_lines.append("=" * 60)
        return "\n".join(context_lines)

    def _get_graph_schema_for_agent(self) -> str:
        """Helper method to get graph schema"""
        try:
            return str(self.graph.get_schema)
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def _add_to_memory(self, user_query: str, response: str, context: Dict):
        """Add conversation to memory, keeping only last 5"""
        memory_entry = {
            "user_query": user_query,
            "response": response,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        self.conversation_memory.append(memory_entry)

        # Keep only last 5 conversations
        if len(self.conversation_memory) > self.max_memory_length:
            self.conversation_memory = self.conversation_memory[
                -self.max_memory_length :
            ]

    def _get_conversation_context(self) -> str:
        """Format conversation memory for agents"""
        if not self.conversation_memory:
            return "No previous conversation history."

        context_lines = ["Recent Conversation History (Last 5 exchanges):"]
        context_lines.append("=" * 60)

        for i, entry in enumerate(self.conversation_memory, 1):
            context_lines.append(f"\n**Exchange {i}:**")
            context_lines.append(f"User: {entry['user_query']}")
            context_lines.append(
                f"Assistant: {entry['response'][:200]}..."
            )  # Truncate long responses

            # Add important context
            if entry["context"].get("products_mentioned"):
                context_lines.append(
                    f"Products mentioned: {', '.join(entry['context']['products_mentioned'])}"
                )
            if entry["context"].get("receipt_used"):
                context_lines.append(
                    f"Receipt accessed: {entry['context']['receipt_used']}"
                )

        context_lines.append("=" * 60)
        return "\n".join(context_lines)

    def _extract_context_from_response(
        self, graph_data: str, receipt_path: str
    ) -> Dict:
        """Extract context information for memory"""
        context = {"products_mentioned": [], "receipt_used": receipt_path}

        # Try to extract product names from graph data
        try:
            import json

            if graph_data:
                data = json.loads(graph_data)
                for item in data:
                    if "name" in item:
                        context["products_mentioned"].append(item["name"])
                    elif "p.name" in item:
                        context["products_mentioned"].append(item["p.name"])
        except:
            pass

        return context

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []

    """Hybrid RAG orchestrator with AI-powered agents"""

    def __init__(
        self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_key: str
    ):
        os.environ["OPENAI_API_KEY"] = openai_key

        # Get the absolute path of the application root directory
        self.app_root = os.path.abspath(os.path.dirname(__file__))

        # Initialize Neo4j Graph Database
        self.graph = Neo4jGraph(
            url=neo4j_uri, username=neo4j_user, password=neo4j_password
        )

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings()
        self.llm = LLM(model="gpt-4o", temperature=0.3)

        # Vector store for PDF receipts
        self.vector_store = None
        self.pdf_metadata = {}

        # Conversation memory - store last 5 conversations
        self.conversation_memory = []
        self.max_memory_length = 5

        # Initialize CrewAI agents
        self._setup_agents()

    def _setup_agents(self):
        """Setup AI-powered CrewAI agents"""

        # 1. Query Analysis Agent
        self.query_analyst = Agent(
            role="Query Analysis Specialist",
            goal="Deeply understand user queries and determine the optimal retrieval strategy",
            backstory="""You are an expert at natural language understanding and query intent analysis.
            You can identify whether a query needs:
            - Product information (prices, specs, features, manufacturers)
            - Receipt/invoice data (warranties, payments, serial numbers)
            - Both types of information (hybrid queries)
            - General system information
            
            You also identify specific products mentioned in queries, even when users use
            informal names or partial descriptions. You understand context and can infer
            what the user is really asking for.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # 2. Graph RAG Agent with Cypher expertise
        self.graph_expert = Agent(
            role="Graph Database Expert",
            goal="Retrieve precise information from Neo4j using intelligent Cypher queries",
            backstory="""You are a Neo4j expert who writes efficient, accurate Cypher queries.
            You understand graph databases deeply and can:
            - Write complex Cypher queries to retrieve product information
            - Navigate relationships between products, manufacturers, and categories
            - Extract specific data points requested by users
            - Handle fuzzy matching and partial product names
            
            You always construct queries that will return meaningful results. 
            You explain what data you found.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # 3. Vector RAG Agent with semantic search expertise
        self.vector_expert = Agent(
            role="Document Search Specialist",
            goal="Find relevant receipt and invoice information using semantic search",
            backstory="""You are an expert at semantic search and document retrieval.
            You excel at:
            - Understanding what information users need from receipts
            - Interpreting receipt content and extracting key details
            - Finding warranty information, payment details, serial numbers, and dates
            
            You know how to search for specific information in unstructured PDF documents
            and present findings in a clear, organized manner.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # 4. Response Coordinator Agent
        self.coordinator = Agent(
            role="Response Coordinator",
            goal="Synthesize information from multiple sources into comprehensive, natural answers",
            backstory="""You are excellent at combining information from different sources
            and presenting it in a user-friendly way. You:
            - Integrate product data from graph databases with receipt information
            - Create coherent narratives from multiple data points
            - Highlight the most relevant information for each query
            - Format responses in a clear, structured manner
            - Indicate which sources provided which information
            
            You never make up information - you only use data provided by other agents.
            When information is missing, you clearly state what's unavailable.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

    def setup_graph_schema(self):
        """Initialize Neo4j graph schema"""
        try:
            self.graph.query("""
                CREATE CONSTRAINT product_id IF NOT EXISTS
                FOR (p:Product) REQUIRE p.id IS UNIQUE
            """)

            self.graph.query("""
                CREATE CONSTRAINT receipt_id IF NOT EXISTS
                FOR (r:Receipt) REQUIRE r.id IS UNIQUE
            """)
            return True, "Graph schema initialized successfully"
        except Exception as e:
            return False, f"Error initializing schema: {str(e)}"

    def add_product_to_graph(self, product_data: Dict):
        """Add product and receipt metadata to Neo4j"""
        try:
            cypher_query = """
            MERGE (p:Product {id: $product_id})
            SET p.name = $name,
                p.category = $category,
                p.price = $price,
                p.manufacturer = $manufacturer,
                p.specs = $specs
            
            WITH p
            MERGE (r:Receipt {id: $receipt_id})
            SET r.path = $receipt_path,
                r.product_id = $product_id
            
            MERGE (p)-[:HAS_RECEIPT]->(r)
            
            WITH p
            MERGE (m:Manufacturer {name: $manufacturer})
            MERGE (p)-[:MANUFACTURED_BY]->(m)
            
            WITH p
            MERGE (c:Category {name: $category})
            MERGE (p)-[:BELONGS_TO]->(c)
            """

            self.graph.query(
                cypher_query,
                {
                    "product_id": product_data["id"],
                    "name": product_data["name"],
                    "category": product_data["category"],
                    "price": product_data["price"],
                    "manufacturer": product_data["manufacturer"],
                    "specs": json.dumps(product_data["specs"]),
                    "receipt_path": product_data["receipt_path"],
                    "receipt_id": f"receipt_{product_data['id']}",
                },
            )

            self.pdf_metadata[product_data["id"]] = product_data["receipt_path"]
            return True, f"Added product {product_data['name']} to graph"
        except Exception as e:
            return False, f"Error adding product: {str(e)}"

    def load_pdf_to_vector_store(self, pdf_path: str, product_id: str):
        """Load PDF receipt and create embeddings"""
        try:
            # Resolve to absolute path
            pdf_path = self._resolve_receipt_path(pdf_path)

            if not os.path.exists(pdf_path):
                return False, f"PDF not found at {pdf_path}"

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["product_id"] = product_id
                doc.metadata["source"] = pdf_path

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db",
                )
            else:
                self.vector_store.add_documents(splits)

            return True, f"Loaded PDF {pdf_path} into vector store"
        except Exception as e:
            return False, f"Error loading PDF: {str(e)}"

    def process_query_with_agents(self, user_query: str) -> Dict:
        """Process query using AI agents - with intelligent receipt loading and memory"""

        # Get available products first
        all_products = self._get_all_products_for_agent()
        graph_schema = self._get_graph_schema_for_agent()

        # Get conversation context
        conversation_context = self._get_conversation_context()

        # Task 1: Analyze the query WITH conversation context
        analysis_task = Task(
            description=f"""Analyze this user query considering previous conversation context.
            
            Current query: "{user_query}"
            
            {conversation_context}
            
            Available products in the system:
            {all_products}
            
            Important Instructions:
            1. Check if the query references previous conversation:
               - Words like "it", "that", "last", "previous", "the product", "same one"
               - If yes, identify what product/topic was discussed before
            
            2. Determine what information is needed:
               - Product information from graph (prices, specs, manufacturers)?
               - Receipt/invoice information (warranties, payment methods, serial numbers, invoice IDs)?
               - Follow-up details about previously mentioned products?
            
            3. If this is a follow-up question:
               - Identify which product from conversation history is being referenced
               - Determine what new information is being requested
            
            4. Identify any specific products mentioned (by name or reference)
            
            Provide analysis:
            - Is this a follow-up question? (yes/no)
            - Product referenced: (from current query or previous context)
            - Query type: "product", "receipt", or "hybrid"
            - Information needed: specific data points
            """,
            agent=self.query_analyst,
            expected_output="Analysis considering conversation context",
        )

        # Task 2: Retrieve graph data AND check for receipt availability
        graph_task = Task(
            description=f"""Retrieve product information from Neo4j, using conversation context if needed.
            
            Current query: "{user_query}"
            
            {conversation_context}
            
            Available Products:
            {all_products}
            
            Instructions:
            1. Review the query analysis to understand what product is being referenced
            2. If it's a follow-up question, use the product from previous conversation
            3. Write a Cypher query to get product details AND receipt path

            For a SPECIFIC product lookup, use this pattern:
            ```cypher
            MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS toLower('product_name') OR p.id = 'product_id'
            OPTIONAL MATCH (p)-[:HAS_RECEIPT]->(r:Receipt)
            RETURN p.id, p.name, p.price, p.specs, p.manufacturer, p.category, r.path as receipt_path
            ```

            For a GENERAL LISTING query (e.g. "top 10 products", "all products", "list products"), use:
            ```cypher
            MATCH (p:Product)
            RETURN p.id as id, p.name as name, p.category as category, p.price as price
            ORDER BY p.name
            LIMIT 10
            ```

            Always write a Cypher query — only respond "NOT_NEEDED" if the query has nothing to do with products at all.

            Put your query in ```cypher``` code blocks.
            """,
            agent=self.graph_expert,
            expected_output="Cypher query to get product and receipt info, or NOT_NEEDED",
        )

        # Task 3: Determine if receipt search is needed
        vector_task = Task(
            description=f"""Determine if we need to search receipt documents, considering conversation context.
            
            Current query: "{user_query}"
            
            {conversation_context}
            
            Query Analysis: Will be provided
            Graph Results: Will be provided
            
            Instructions:
            1. Check if the user is asking for information typically in receipts:
               - Invoice ID, receipt number, order number
               - Payment method, transaction details
               - Warranty information, warranty period
               - Serial numbers
               - Purchase dates, order dates
               - Tax information, total amount paid
               - Return policy, refund information
               - Shipping details
            
            2. Consider follow-up questions like:
               - "What about the warranty?" (referring to previously mentioned product)
               - "Show me the invoice" (for last discussed item)
            
            3. If receipt search is needed, respond:
               "SEARCH_NEEDED: <describe what to search for>"
            
            4. Otherwise respond: "NOT_NEEDED"
            """,
            agent=self.vector_expert,
            expected_output="SEARCH_NEEDED with description or NOT_NEEDED",
        )

        # Create and run the crew for analysis and planning
        planning_crew = Crew(
            agents=[self.query_analyst, self.graph_expert, self.vector_expert],
            tasks=[analysis_task, graph_task, vector_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            # Run planning phase
            planning_crew.kickoff()

            # Execute graph query and get receipt path
            graph_output = str(graph_task.output)
            vector_output = str(vector_task.output)

            graph_data = None
            receipt_path = None
            product_id = None

            if "NOT_NEEDED" not in graph_output:
                import re

                cypher_pattern = r"```cypher\s*(.*?)\s*```"
                matches = re.findall(
                    cypher_pattern, graph_output, re.DOTALL | re.IGNORECASE
                )

                if matches:
                    cypher_query = matches[0].strip()
                    graph_data_raw = self._query_graph_for_agent(cypher_query)
                    graph_data = graph_data_raw

                    # Extract receipt path and product ID from results
                    try:
                        import json

                        results = json.loads(graph_data_raw)
                        if results and len(results) > 0:
                            result = results[0]
                            receipt_path_raw = result.get("receipt_path") or result.get(
                                "r.path"
                            )
                            # Resolve to absolute path
                            receipt_path = (
                                self._resolve_receipt_path(receipt_path_raw)
                                if receipt_path_raw
                                else None
                            )
                            product_id = result.get("id") or result.get("p.id")
                    except:
                        pass
                else:
                    # Fallback pattern matching
                    match_pattern = r"(MATCH\s+.*?RETURN.*?)(?=\n\n|\Z|```)"
                    matches = re.findall(
                        match_pattern, graph_output, re.DOTALL | re.IGNORECASE
                    )
                    if matches:
                        cypher_query = matches[0].strip()
                        graph_data_raw = self._query_graph_for_agent(cypher_query)
                        graph_data = graph_data_raw

                        try:
                            import json

                            results = json.loads(graph_data_raw)
                            if results and len(results) > 0:
                                result = results[0]
                                receipt_path_raw = result.get(
                                    "receipt_path"
                                ) or result.get("r.path")
                                # Resolve to absolute path
                                receipt_path = (
                                    self._resolve_receipt_path(receipt_path_raw)
                                    if receipt_path_raw
                                    else None
                                )
                                product_id = result.get("id") or result.get("p.id")
                        except:
                            pass

            # Execute vector search - load receipt on-the-fly if needed
            vector_data = None
            if "SEARCH_NEEDED" in vector_output:
                # Extract search query
                search_match = re.search(
                    r"SEARCH_NEEDED:\s*(.+?)(?:\n|$)", vector_output
                )
                if search_match:
                    search_description = search_match.group(1).strip()
                else:
                    search_description = user_query

                # If we have a receipt path, load it on-the-fly
                if receipt_path:
                    vector_data = self._search_vector_for_agent(
                        query=user_query,
                        product_id=product_id,
                        receipt_path=receipt_path,
                    )
                else:
                    # Search existing vector store
                    vector_data = self._search_vector_for_agent(
                        query=user_query, product_id=product_id
                    )

            # Task 4: Synthesize natural response with actual data AND context
            synthesis_task = Task(
                description=f"""Create a natural, conversational response considering the full conversation context.
                
                Current query: "{user_query}"
                
                {conversation_context}
                
                Query Analysis: {str(analysis_task.output)}
                
                Retrieved Data:
                - Graph Data (Product info): {graph_data if graph_data else "No specific graph data retrieved"}
                - All Available Products (use this if graph data is missing): {all_products}
                - Receipt Data: {vector_data if vector_data else "No receipt data needed"}
                - Receipt was loaded from: {receipt_path if receipt_path else "Not applicable"}
                
                Instructions:
                1. Consider the conversation history when formulating your response
                2. If this is a follow-up question, reference the previous context naturally
                3. Extract the specific information the user asked for
                4. Present it naturally and conversationally
                5. If information is not found, say "No information found for [specific query]"
                
                Examples of context-aware responses:
                - "The Plant Pro Max you asked about earlier costs $149.99"
                - "For that product, the warranty period is 2 years"
                - "The invoice ID for the Smartphone Plus we discussed is INV-2024-0512"
                - "Looking at the same product, the serial number is SN-2024-XYZ"
                
                Be conversational and acknowledge the conversation flow naturally.
                """,
                agent=self.coordinator,
                expected_output="Natural conversational response with context awareness",
            )

            synthesis_crew = Crew(
                agents=[self.coordinator],
                tasks=[synthesis_task],
                process=Process.sequential,
                verbose=True,
            )

            result = synthesis_crew.kickoff()

            # Extract context for memory
            context_info = self._extract_context_from_response(graph_data, receipt_path)

            # Add to conversation memory
            self._add_to_memory(user_query, str(result), context_info)

            # Determine sources used
            sources_used = []
            if graph_data:
                sources_used.append("Graph RAG (Neo4j)")
            if vector_data:
                sources_used.append("Vector RAG (PDF Embeddings)")
            if not sources_used:
                sources_used.append("System")

            return {
                "response": str(result),
                "sources": sources_used,
                "receipt_info": {"path": receipt_path, "product_id": product_id}
                if receipt_path
                else None,
                "debug_info": {
                    "conversation_context": conversation_context,
                    "analysis": str(analysis_task.output),
                    "graph_reasoning": graph_output,
                    "graph_data": graph_data,
                    "receipt_path_used": receipt_path,
                    "vector_reasoning": vector_output,
                    "vector_data": vector_data,
                    "context_extracted": context_info,
                },
            }

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "sources": ["Error"],
                "debug_info": {"error": error_details},
            }


# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI-Powered Hybrid RAG", page_icon="🤖", layout="wide"
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .source-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .graph-badge {
            background-color: #3b82f6;
            color: white;
        }
        .vector-badge {
            background-color: #10b981;
            color: white;
        }
        .system-badge {
            background-color: #6b7280;
            color: white;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("🤖 AI-Powered Hybrid RAG System")
    st.markdown(
        "**Intelligent agents make decisions using LLMs - No hardcoded rules!**"
    )

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Debug mode toggle
        st.session_state.debug_mode = st.toggle(
            "🐛 Debug Mode",
            value=st.session_state.get("debug_mode", False),
            help="Show detailed agent reasoning and data",
        )

        st.divider()

        connection_type = st.radio(
            "Database Type",
            ["Local Neo4j", "Neo4j Aura DB (Cloud)"],
            help="Select your Neo4j deployment type",
        )

        with st.expander("🔐 Database Credentials", expanded=False):
            if connection_type == "Neo4j Aura DB (Cloud)":
                st.info("💡 Aura DB uses encrypted connections (neo4j+s://)")
                neo4j_uri = st.text_input(
                    "Neo4j Aura URI",
                    value=os.getenv("NEO4J_URI", "neo4j+s://xxxxx.databases.neo4j.io"),
                    help="Get this from your Aura DB console",
                )
                st.caption("⚠️ URI must start with neo4j+s://")
            else:
                neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))

            neo4j_user = st.text_input("Neo4j User", value=os.getenv("NEO4J_USER", "neo4j"))
            neo4j_password = st.text_input("Neo4j Password", type="password", value=os.getenv("NEO4J_PASSWORD", ""))
            openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

            if (
                connection_type == "Neo4j Aura DB (Cloud)"
                and neo4j_uri
                and not neo4j_uri.startswith("neo4j+s://")
            ):
                st.warning("⚠️ Aura DB URIs should start with 'neo4j+s://'")

        st.divider()

        # Initialize system
        if "rag_system" not in st.session_state:
            if st.button("🚀 Initialize AI System", type="primary"):
                if not openai_key:
                    st.error("Please provide OpenAI API Key")
                else:
                    with st.spinner("Initializing AI agents..."):
                        try:
                            st.session_state.rag_system = HybridRAGSystem(
                                neo4j_uri, neo4j_user, neo4j_password, openai_key
                            )
                            success, msg = (
                                st.session_state.rag_system.setup_graph_schema()
                            )
                            if success:
                                st.success("✅ " + msg)
                                st.success("✅ AI Agents initialized")
                                st.session_state.initialized = True
                            else:
                                st.error(msg)
                        except Exception as e:
                            st.error(f"Initialization failed: {str(e)}")
        else:
            st.success("✅ AI System Active")

            if st.button("🔄 Reset System"):
                del st.session_state.rag_system
                st.session_state.initialized = False
                st.rerun()

        st.divider()

        # Product Management
        if st.session_state.get("initialized", False):
            st.header("📦 Product Management")

            with st.expander("➕ Add Product"):
                product_id = st.text_input("Product ID", key="prod_id")
                product_name = st.text_input("Product Name", key="prod_name")
                category = st.selectbox(
                    "Category", ["Electronics", "Accessories", "Clothing", "Other"]
                )
                price = st.number_input(
                    "Price ($)", min_value=0.0, step=0.01, key="prod_price"
                )
                manufacturer = st.text_input("Manufacturer", key="prod_manu")

                st.write("**Specifications (JSON)**")
                specs_text = st.text_area(
                    "Specs", value='{"key": "value"}', key="prod_specs"
                )

                receipt_path = st.text_input("Receipt PDF Path", key="receipt_path")

                if st.button("Add Product to Graph"):
                    try:
                        specs = json.loads(specs_text)
                        product_data = {
                            "id": product_id,
                            "name": product_name,
                            "category": category,
                            "price": price,
                            "manufacturer": manufacturer,
                            "specs": specs,
                            "receipt_path": receipt_path,
                        }
                        success, msg = st.session_state.rag_system.add_product_to_graph(
                            product_data
                        )
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in specifications")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with st.expander("📄 Load Receipt PDF"):
                pdf_file = st.file_uploader("Upload Receipt PDF", type=["pdf"])
                load_product_id = st.text_input(
                    "Product ID for this receipt", key="load_prod_id"
                )

                if st.button("Load PDF to Vector Store"):
                    if pdf_file and load_product_id:
                        pdf_path = f"./temp_{pdf_file.name}"
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_file.getbuffer())

                        success, msg = (
                            st.session_state.rag_system.load_pdf_to_vector_store(
                                pdf_path, load_product_id
                            )
                        )

                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please provide both PDF and Product ID")

        st.divider()
        st.markdown("### 🤖 AI Agents")
        st.info(
            "**Query Analyst:** Understands intent\n\n"
            "**Graph Expert:** Writes Cypher queries\n\n"
            "**Vector Expert:** Searches receipts\n\n"
            "**Coordinator:** Synthesizes responses"
        )

        st.divider()
        st.markdown("### 💭 Memory")
        if st.session_state.get("initialized", False):
            memory_count = len(st.session_state.rag_system.conversation_memory)
            st.info(f"📝 Remembering last {memory_count}/5 conversations")

            if st.button("🗑️ Clear Memory"):
                st.session_state.rag_system.clear_memory()
                st.success("Memory cleared!")
                st.rerun()
        else:
            st.info("📝 Memory: Not active")

    # Main Content
    if not st.session_state.get("initialized", False):
        st.info("👈 Please initialize the AI system from the sidebar")

        st.markdown("### 🎯 What Makes This Different?")
        st.markdown("""
        This system uses **AI agents that think and decide** - no hardcoded rules!
        
        - 🧠 **Query Analyst** understands your intent using natural language
        - 📊 **Graph Expert** writes custom Cypher queries for each question
        - 📄 **Vector Expert** intelligently searches receipts semantically
        - 🎨 **Coordinator** creates natural, contextual responses
        
        The agents collaborate and make autonomous decisions based on your question!
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💬 Example Queries")
            st.code("""
• What laptops do you have?
• Show me the warranty for my laptop
• How much did I pay for the headphones?
• What products does Apple make?
• When did I buy the keyboard?
            """)

        with col2:
            st.markdown("### 🤖 AI Decision Making")
            st.code("""
Agents automatically:
✓ Understand query intent
✓ Write custom Cypher queries
✓ Search receipts semantically
✓ Combine multiple sources
✓ Generate natural responses
            """)

    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    # Show sources
                    # st.markdown("**Sources Used:**")
                    # sources_html = ""
                    # for source in message["metadata"]["sources"]:
                    #     if "Graph" in source:
                    #         sources_html += '<span class="source-badge graph-badge">🗄️ Graph RAG</span>'
                    #     elif "Vector" in source:
                    #         sources_html += '<span class="source-badge vector-badge">📄 Vector RAG</span>'
                    #     else:
                    #         sources_html += '<span class="source-badge system-badge">ℹ️ System</span>'
                    # st.markdown(sources_html, unsafe_allow_html=True)

                    # Show receipt action buttons if receipt was accessed
                    if message["metadata"].get("receipt_info"):
                        receipt_info = message["metadata"]["receipt_info"]
                        receipt_path = receipt_info.get("path")

                        if receipt_path and os.path.exists(receipt_path):
                            st.markdown("---")
                            st.markdown("**📄 Receipt Actions:**")

                            col1, col2 = st.columns(2)

                            with col1:
                                # Download button
                                try:
                                    with open(receipt_path, "rb") as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                        filename = os.path.basename(receipt_path)
                                        st.download_button(
                                            label="⬇️ Download Receipt",
                                            data=pdf_bytes,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_{message.get('timestamp', 0)}",
                                        )
                                except Exception as e:
                                    st.error(f"Cannot access receipt: {str(e)}")

                            with col2:
                                # Preview button
                                if st.button(
                                    "👁️ Preview Receipt",
                                    key=f"preview_{message.get('timestamp', 0)}",
                                ):
                                    st.session_state[
                                        f"show_preview_{message.get('timestamp', 0)}"
                                    ] = True

                            # Show preview if button was clicked
                            if st.session_state.get(
                                f"show_preview_{message.get('timestamp', 0)}", False
                            ):
                                with st.expander("📄 Receipt Preview", expanded=True):
                                    try:
                                        # Display PDF using iframe
                                        with open(receipt_path, "rb") as pdf_file:
                                            base64_pdf = base64.b64encode(
                                                pdf_file.read()
                                            ).decode("utf-8")
                                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                                            st.markdown(
                                                pdf_display, unsafe_allow_html=True
                                            )

                                            # Close button
                                            if st.button(
                                                "Close Preview",
                                                key=f"close_{message.get('timestamp', 0)}",
                                            ):
                                                st.session_state[
                                                    f"show_preview_{message.get('timestamp', 0)}"
                                                ] = False
                                                st.rerun()
                                    except Exception as e:
                                        st.error(f"Cannot preview receipt: {str(e)}")
                                        # Fallback: show text content
                                        try:
                                            from PyPDFLoader import PyPDFLoader

                                            loader = PyPDFLoader(receipt_path)
                                            documents = loader.load()
                                            st.text_area(
                                                "Receipt Content",
                                                "\n\n".join(
                                                    [
                                                        doc.page_content
                                                        for doc in documents
                                                    ]
                                                ),
                                                height=400,
                                            )
                                        except:
                                            st.error("Unable to load receipt content")

                    # Show debug info only if debug mode is enabled
                    if st.session_state.get("debug_mode", False) and message[
                        "metadata"
                    ].get("debug_info"):
                        with st.expander("🐛 Debug: Agent Thinking Process"):
                            debug = message["metadata"]["debug_info"]

                            if debug.get("analysis"):
                                st.markdown("**🔍 Query Analysis:**")
                                st.text_area(
                                    "Analysis",
                                    debug["analysis"],
                                    height=100,
                                    key=f"analysis_{message.get('timestamp', 0)}",
                                    disabled=True,
                                )

                            if debug.get("graph_reasoning"):
                                st.markdown("**📊 Graph Expert Reasoning:**")
                                st.text_area(
                                    "Reasoning",
                                    debug["graph_reasoning"],
                                    height=100,
                                    key=f"graph_reason_{message.get('timestamp', 0)}",
                                    disabled=True,
                                )

                            if debug.get("graph_data"):
                                st.markdown("**📊 Graph Data Retrieved:**")
                                st.code(debug["graph_data"], language="json")

                            if debug.get("receipt_path_used"):
                                st.markdown("**📄 Receipt Loaded On-The-Fly:**")
                                st.info(
                                    f"Dynamically loaded: `{debug['receipt_path_used']}`"
                                )

                            if debug.get("conversation_context"):
                                st.markdown("**💭 Conversation Context Used:**")
                                st.text_area(
                                    "Context",
                                    debug["conversation_context"],
                                    height=150,
                                    key=f"context_{message.get('timestamp', 0)}",
                                    disabled=True,
                                )

                            if debug.get("vector_reasoning"):
                                st.markdown("**📄 Vector Expert Reasoning:**")
                                st.text_area(
                                    "Reasoning",
                                    debug["vector_reasoning"],
                                    height=100,
                                    key=f"vector_reason_{message.get('timestamp', 0)}",
                                    disabled=True,
                                )

                            if debug.get("vector_data"):
                                st.markdown("**📄 Vector Data Retrieved:**")
                                st.text_area(
                                    "Data",
                                    debug["vector_data"],
                                    height=150,
                                    key=f"vector_data_{message.get('timestamp', 0)}",
                                    disabled=True,
                                )

        # Chat input
        if prompt := st.chat_input("Ask me anything about products or receipts..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI Agents are thinking..."):
                    result = st.session_state.rag_system.process_query_with_agents(
                        prompt
                    )

                    st.markdown(result["response"])

                    # Show sources
                    # st.markdown("**Sources Used:**")
                    # sources_html = ""
                    # for source in result["sources"]:
                    #     if "Graph" in source:
                    #         sources_html += '<span class="source-badge graph-badge">🗄️ Graph RAG</span>'
                    #     elif "Vector" in source:
                    #         sources_html += '<span class="source-badge vector-badge">📄 Vector RAG</span>'
                    #     else:
                    #         sources_html += '<span class="source-badge system-badge">ℹ️ System</span>'
                    # st.markdown(sources_html, unsafe_allow_html=True)

                    # Show receipt action buttons if receipt was accessed
                    if result.get("receipt_info"):
                        receipt_info = result["receipt_info"]
                        receipt_path = receipt_info.get("path")

                        if receipt_path and os.path.exists(receipt_path):
                            st.markdown("---")
                            st.markdown("**📄 Receipt Actions:**")

                            col1, col2 = st.columns(2)

                            with col1:
                                # Download button
                                try:
                                    with open(receipt_path, "rb") as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                        filename = os.path.basename(receipt_path)
                                        st.download_button(
                                            label="⬇️ Download Receipt",
                                            data=pdf_bytes,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key="download_current",
                                        )
                                except Exception as e:
                                    st.error(f"Cannot access receipt: {str(e)}")

                            with col2:
                                # Preview button
                                if st.button(
                                    "👁️ Preview Receipt", key="preview_current"
                                ):
                                    st.session_state["show_preview_current"] = True

                            # Show preview if button was clicked
                            if st.session_state.get("show_preview_current", False):
                                with st.expander("📄 Receipt Preview", expanded=True):
                                    try:
                                        # Display PDF using iframe
                                        with open(receipt_path, "rb") as pdf_file:
                                            base64_pdf = base64.b64encode(
                                                pdf_file.read()
                                            ).decode("utf-8")
                                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                                            st.markdown(
                                                pdf_display, unsafe_allow_html=True
                                            )

                                            # Close button
                                            if st.button(
                                                "Close Preview", key="close_current"
                                            ):
                                                st.session_state[
                                                    "show_preview_current"
                                                ] = False
                                                st.rerun()
                                    except Exception as e:
                                        st.error(f"Cannot preview receipt: {str(e)}")
                                        # Fallback: show text content
                                        try:
                                            loader = PyPDFLoader(receipt_path)
                                            documents = loader.load()
                                            st.text_area(
                                                "Receipt Content",
                                                "\n\n".join(
                                                    [
                                                        doc.page_content
                                                        for doc in documents
                                                    ]
                                                ),
                                                height=400,
                                                disabled=True,
                                            )
                                        except:
                                            st.error("Unable to load receipt content")

                    # Show debug info only if debug mode is enabled
                    if st.session_state.get("debug_mode", False) and result.get(
                        "debug_info"
                    ):
                        with st.expander("🐛 Debug: Agent Thinking Process"):
                            debug = result["debug_info"]

                            if debug.get("analysis"):
                                st.markdown("**🔍 Query Analysis:**")
                                st.text_area(
                                    "Analysis",
                                    debug["analysis"],
                                    height=100,
                                    key="current_analysis",
                                    disabled=True,
                                )

                            if debug.get("graph_reasoning"):
                                st.markdown("**📊 Graph Expert Reasoning:**")
                                st.text_area(
                                    "Reasoning",
                                    debug["graph_reasoning"],
                                    height=100,
                                    key="current_graph_reason",
                                    disabled=True,
                                )

                            if debug.get("graph_data"):
                                st.markdown("**📊 Graph Data Retrieved:**")
                                st.code(debug["graph_data"], language="json")

                            if debug.get("receipt_path_used"):
                                st.markdown("**📄 Receipt Loaded On-The-Fly:**")
                                st.info(
                                    f"Dynamically loaded: `{debug['receipt_path_used']}`"
                                )

                            if debug.get("vector_reasoning"):
                                st.markdown("**📄 Vector Expert Reasoning:**")
                                st.text_area(
                                    "Reasoning",
                                    debug["vector_reasoning"],
                                    height=100,
                                    key="current_vector_reason",
                                    disabled=True,
                                )

                            if debug.get("vector_data"):
                                st.markdown("**📄 Vector Data Retrieved:**")
                                st.text_area(
                                    "Data",
                                    debug["vector_data"],
                                    height=150,
                                    key="current_vector_data",
                                    disabled=True,
                                )

                    # Add to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["response"],
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {
                                "sources": result["sources"],
                                "receipt_info": result.get("receipt_info"),
                                "debug_info": result.get("debug_info", {}),
                            },
                        }
                    )

        # Quick actions
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 What products exist?"):
                query = "What products are available in the system?"
                st.session_state.messages.append({"role": "user", "content": query})
                result = st.session_state.rag_system.process_query_with_agents(query)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["response"],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "sources": result["sources"],
                            "debug_info": result.get("debug_info", {}),
                        },
                    }
                )
                st.rerun()

        with col2:
            if st.button("💡 How does this work?"):
                query = "Explain how this hybrid RAG system works"
                st.session_state.messages.append({"role": "user", "content": query})
                result = st.session_state.rag_system.process_query_with_agents(query)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["response"],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "sources": result["sources"],
                            "receipt_info": result.get("receipt_info"),
                            "agent_outputs": result.get("agent_outputs", {}),
                        },
                    }
                )
                st.rerun()

        with col3:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                if st.session_state.get("initialized", False):
                    st.session_state.rag_system.clear_memory()
                st.rerun()


if __name__ == "__main__":
    main()
