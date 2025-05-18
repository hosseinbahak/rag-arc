# rag/core.py - Improved version
from typing import Dict, List, Optional, Any
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import traceback
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EquipmentInspectionRAG")

class EquipmentInspectionRAG:
    def __init__(self, excel_path: str):
        logger.info(f"Initializing EquipmentInspectionRAG with Excel path: {excel_path}")
        self.excel_path = excel_path
        self.df = None
        self.vectorstore = None
        self.retriever = None
        
        # Ensure OPENAI_API_KEY is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        try:
            # Using gpt-4-turbo-preview for potentially better Farsi handling and performance
            self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, api_key=openai_api_key)
            logger.info("ChatOpenAI model initialized.")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI: {e}")
            self.llm = None  # Set to None if initialization fails

        try:
            self.setup()  # Initial data load and vectorstore creation
            logger.info("Setup complete.")
        except Exception as e:
            logger.error(f"Error during initial setup: {e}")
            traceback.print_exc()  # Print full traceback
            self.df = pd.DataFrame()  # Ensure df is a DataFrame even if setup fails
            self.vectorstore = None
            self.retriever = None

        self.graph = None  # Graph will be created on first query

    def setup(self):
        """Loads data from Excel, prepares documents, and initializes vectorstore."""
        logger.info(f"Attempting to load data from {self.excel_path}...")
        try:
            # Read, convert all to string to avoid type issues, fill NaN
            self.df = pd.read_excel(self.excel_path)
            logger.info("Excel file read successfully.")
            
            # Important: Convert *all* columns to string and fillna AFTER reading
            # This prevents errors with mixed types or NaNs during string operations later
            for col in self.df.columns:
                self.df[col] = self.df[col].astype(str).fillna("None")
            
            logger.info(f"DataFrame loaded and cleaned. Found {len(self.df)} rows.")
            if len(self.df) == 0:
                logger.warning("DataFrame is empty after loading.")

            logger.info("Creating documents for vectorstore...")
            documents = []

            # Define columns to include in the RAG document content and metadata
            # Use a dictionary to map desired metadata key to Excel column name
            metadata_map = {
                "punch_id": 'Punch ID',
                "task_number": 'Task Number',
                "item": 'Item',
                "item_type": 'ItemType',
                "discipline": 'Disc',
                "form_type": 'Form Type',
                "punch_status": 'Punch Status',
                "punch_type": 'Type Of Punch'
            }
            
            content_cols = [
                'Punch ID', 'Task Number', 'Punch', 'Type Of Punch',
                'Punch Status', 'RevisionDate', 'RevisionNumber',
                'Item', 'ItemType', 'Disc', 'Form Type', 'PrvPunchId'
            ]

            # Ensure all content_cols exist, handle missing columns gracefully
            existing_content_cols = [col for col in content_cols if col in self.df.columns]
            missing_cols = [col for col in content_cols if col not in self.df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns in Excel for RAG content: {missing_cols}. These will be omitted.")

            # Ensure all metadata columns exist
            existing_metadata_map = {k: v for k, v in metadata_map.items() if v in self.df.columns}
            missing_metadata_cols = [v for k, v in metadata_map.items() if v not in self.df.columns]
            if missing_metadata_cols:
                logger.warning(f"Missing expected columns in Excel for metadata: {missing_metadata_cols}. Metadata may be incomplete.")

            for index, row in self.df.iterrows():
                # Construct content string from existing columns
                content_parts = [f"{col}: {row[col]}" for col in existing_content_cols]
                content = "\n".join(content_parts)

                # Construct metadata dictionary from existing columns
                metadata = {"_original_index": index}  # Always add original index

                for key, col_name in existing_metadata_map.items():
                    metadata[key] = row[col_name]

                documents.append(Document(page_content=content, metadata=metadata))

            logger.info(f"Created {len(documents)} documents for vectorstore.")
            if len(documents) == 0 and len(self.df) > 0:
                logger.error("No documents created despite having rows in DataFrame. Check column names used for content.")

            logger.info("Initializing Chroma vectorstore...")
            if documents:  # Only attempt to create vectorstore if there are documents
                try:
                    # Consider using a persistent directory for production use
                    self.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=OpenAIEmbeddings()  # Ensure OPENAI_API_KEY is set for this too
                    )
                    logger.info("Vectorstore initialized.")
                    self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    logger.info("Retriever created.")
                except Exception as e:
                    logger.error(f"Error initializing Chroma or Embeddings: {e}")
                    traceback.print_exc()
                    self.vectorstore = None
                    self.retriever = None
            else:
                logger.warning("No documents created, skipping vectorstore initialization.")
                self.vectorstore = None
                self.retriever = None

        except FileNotFoundError:
            logger.error(f"Excel file not found at {self.excel_path}")
            self.df = pd.DataFrame()  # Initialize empty DataFrame to prevent errors
            self.vectorstore = None
            self.retriever = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during setup: {e}")
            traceback.print_exc()  # Print full traceback
            self.df = pd.DataFrame()  # Initialize empty DataFrame
            self.vectorstore = None
            self.retriever = None

    def create_graph(self):
        """Defines and compiles the LangGraph workflow."""
        logger.info("Attempting to create RAG graph...")
        if not self.llm or not self.retriever:
            logger.error("Cannot create graph: LLM or Retriever not initialized.")
            self.graph = None
            return self.graph

        class State(Dict):
            query: str
            retrieved_documents: Optional[List[Document]] = None  # Specify type hint
            context: Optional[str] = None
            answer: Optional[str] = None

        def retrieve(state: State) -> State:
            logger.info(f"Graph node 'retrieve' called for query: {state['query'][:50]}...")
            if not self.retriever:
                logger.warning("Retriever not initialized in retrieve node.")
                state["retrieved_documents"] = []
                return state

            try:
                docs = self.retriever.get_relevant_documents(state["query"])
                state["retrieved_documents"] = docs
                logger.info(f"Retrieved {len(docs)} documents.")
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}")
                traceback.print_exc()
                state["retrieved_documents"] = []

            return state

        def generate_context(state: State) -> State:
            logger.info("Graph node 'generate_context' called.")
            if not state["retrieved_documents"]:
                state["context"] = "No relevant documents found in the database."
                logger.warning("No documents to generate context from.")
            else:
                context = "\n\n".join(doc.page_content for doc in state["retrieved_documents"])
                state["context"] = context

            return state

        def generate_answer(state: State) -> State:
            logger.info("Graph node 'generate_answer' called.")
            if not self.llm:
                state["answer"] = "LLM not initialized. Cannot generate answer."
                logger.error("LLM not initialized in generate_answer node.")
                return state

            if not state["context"] or state["context"] == "No relevant documents found in the database.":
                state["answer"] = "Based on the available data, I cannot find relevant information to answer your question."
                logger.warning("No context provided for answer generation.")
                return state

            template = """
            You are an expert assistant for an equipment inspection and punch management system.
            Use the following information from the database context to answer the user's question accurately and concisely.

            Context:
            {context}

            Question: {query}

            Instructions:
            - Answer the question based ONLY on the provided context.
            - If the context does not contain the information to answer the question, respond with "Based on the available data, I cannot answer your question." or similar.
            - Do NOT make up information.
            - Answer in the same language as the question (English or Persian/Farsi).
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            try:
                logger.info("Invoking LLM chain...")
                answer = chain.invoke({"context": state["context"], "query": state["query"]})
                state["answer"] = answer
                logger.info("Answer generated successfully.")
            except Exception as e:
                logger.error(f"Error during LLM chain invocation: {e}")
                traceback.print_exc()
                state["answer"] = "An error occurred while generating the answer."

            return state

        workflow = StateGraph(State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate_context", generate_context)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_edge("retrieve", "generate_context")
        workflow.add_edge("generate_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.set_entry_point("retrieve")
        self.graph = workflow.compile()
        logger.info("RAG graph compiled.")
        return self.graph

    def query(self, question: str) -> str:
        """Runs a query through the RAG graph."""
        logger.info(f"Received query: {question}")
        if not hasattr(self, 'graph') or self.graph is None:
            logger.info("Graph not initialized or failed initialization, attempting to create.")
            self.create_graph()  # Try creating it now if it failed before

        if not self.graph:
            logger.error("RAG graph is not available.")
            return "RAG system is not fully initialized. Cannot process query."

        try:
            logger.info("Invoking RAG graph...")
            result = self.graph.invoke({"query": question})
            logger.info("Graph invocation complete.")
            return result.get("answer", "No answer generated by the graph.")
        except Exception as e:
            logger.error(f"Error during graph invocation: {e}")
            traceback.print_exc()
            return "An error occurred while processing your query."

    def filter_punches(self, disc: str = "", item_type: str = "", punch_status: str = "") -> List[Dict[str, Any]]:
        """Filters the DataFrame based on exact matches for specified columns."""
        logger.info(f"Filtering punches with disc='{disc}', item_type='{item_type}', punch_status='{punch_status}'")
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is not loaded or is empty.")
            return []  # Return empty list if no data

        df = self.df.copy()
        original_count = len(df)

        # Use boolean indexing for exact matches where specified
        # Ensure columns exist AND handle potential None values from fillna appropriately
        # Using .astype(str) again defensively before comparison
        if disc:
            if 'Disc' in df.columns:
                df = df[df['Disc'].astype(str).str.strip().str.lower() == disc.strip().lower()]
                logger.info(f"Filtered by Disc='{disc}', remaining rows: {len(df)}")
            else:
                logger.warning("Warning: 'Disc' column not found for filtering.")

        if item_type:
            if 'ItemType' in df.columns:
                df = df[df['ItemType'].astype(str).str.strip().str.lower() == item_type.strip().lower()]
                logger.info(f"Filtered by ItemType='{item_type}', remaining rows: {len(df)}")
            else:
                logger.warning("Warning: 'ItemType' column not found for filtering.")

        if punch_status:
            if 'Punch Status' in df.columns:
                # Ensure the status we are filtering by exists in the data
                # And handle potential case sensitivity or extra spaces
                df = df[df['Punch Status'].astype(str).str.strip().str.lower() == punch_status.strip().lower()]
                logger.info(f"Filtered by Punch Status='{punch_status}', remaining rows: {len(df)}")
            else:
                logger.warning("Warning: 'Punch Status' column not found for filtering.")

        logger.info(f"Finished filtering. Started with {original_count}, ended with {len(df)}.")
        # Convert DataFrame rows to list of dictionaries
        return df.to_dict(orient="records")

    def add_punch_resolution(self, punch_id: str, new_status: str = "1",
                             resolution_text: str = "", revision: str = "") -> None:
        """Adds a new row representing a resolution/revision for an existing punch."""
        logger.info(f"Received request to add resolution for Punch ID: {punch_id}")
        if self.df is None or self.df.empty:
            logger.error("DataFrame is not loaded or is empty.")
            raise ValueError("DataFrame is not loaded. Cannot add resolution.")

        # We assume 'Punch ID' is the identifier for the punch *thread*.
        # Find the latest revision for this punch ID thread.
        if 'Punch ID' not in self.df.columns or 'RevisionNumber' not in self.df.columns:
            logger.error("'Punch ID' or 'RevisionNumber' column missing in data.")
            raise ValueError("'Punch ID' or 'RevisionNumber' column missing in data. Cannot track resolutions.")

        # Find all rows belonging to the punch thread identified by `punch_id`
        # Use .astype(str).str.strip() for robust comparison
        punch_thread_rows = self.df[self.df['Punch ID'].astype(str).str.strip() == str(punch_id).strip()].copy()

        if punch_thread_rows.empty:
            logger.error(f"Punch thread with ID '{punch_id}' not found in DataFrame.")
            raise ValueError(f"Punch ID {punch_id} not found to add resolution.")

        # Find the latest revision within this thread to base the new row on
        try:
            # Convert RevisionNumber to numeric, coercing errors to NaN
            punch_thread_rows['RevisionNumber_numeric'] = pd.to_numeric(punch_thread_rows['RevisionNumber'], errors='coerce')
            # Sort by numeric revision, placing NaNs (errors) first to ensure a valid number is picked if possible
            latest_revision_row = punch_thread_rows.sort_values(by='RevisionNumber_numeric', ascending=False, na_position='last').iloc[0]
            latest_revision_number = latest_revision_row.get('RevisionNumber_numeric')  # Use .get to be safe
            if pd.isna(latest_revision_number):
                latest_revision_number = -1  # Use a flag if not a valid number
                logger.warning(f"Latest revision number for {punch_id} is not numeric ('{latest_revision_row.get('RevisionNumber', 'N/A')}'). Cannot auto-increment.")
            else:
                latest_revision_number = int(latest_revision_number)  # Convert to int if numeric

        except Exception as e:
            logger.warning(f"Error finding latest revision number for {punch_id}: {e}. Falling back to simple last row and revision '0'.")
            traceback.print_exc()
            # If numerical sort fails, just grab the last row found by ID
            latest_revision_row = punch_thread_rows.iloc[-1].copy()
            latest_revision_number = -1  # Indicate unknown/non-numeric revision

        # Create the new resolution record based on the latest revision row found
        new_record = latest_revision_row.copy()

        # Update fields for the new revision row
        # The new row *is* a revision of the punch thread identified by 'Punch ID'
        new_record['PrvPunchId'] = latest_revision_row['Punch ID']  # Link back to the ID of the previous revision row
        new_record['Punch ID'] = latest_revision_row['Punch ID']  # The new row keeps the same Punch ID (thread ID)
        new_record['Punch Status'] = str(new_status)  # Ensure status is string
        
        # Check if 'Punch.1' column exists before attempting to use it
        if 'Punch.1' in new_record:
            new_record['Punch.1'] = resolution_text  # Add resolution text
        else:
            logger.warning("'Punch.1' column not found for resolution text. Using 'Punch' column instead.")
            if 'Punch' in new_record:
                new_record['Punch'] = resolution_text
            else:
                logger.error("Neither 'Punch.1' nor 'Punch' column found. Resolution text may not be stored.")
        
        new_record['RevisionDate'] = revision  # Use the provided revision identifier (timestamp/string)

        # Increment Revision Number
        if latest_revision_number != -1:
            new_record['RevisionNumber'] = str(latest_revision_number + 1)
            logger.info(f"Incremented revision number for {punch_id} from {latest_revision_number} to {new_record['RevisionNumber']}.")
        else:
            # If previous was non-numeric or unknown, set a default or specific value
            original_rev_str = latest_revision_row.get('RevisionNumber', '0')
            new_record['RevisionNumber'] = f"{original_rev_str}_res_{revision}"  # Example: "RevA_res_2023-10-27"
            logger.info(f"Set revision number for {punch_id} based on original '{original_rev_str}' and new revision text '{revision}'.")

        # Remove the temporary numeric column
        if 'RevisionNumber_numeric' in new_record:
            del new_record['RevisionNumber_numeric']

        # Append the new row to the DataFrame
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)

        logger.info(f"Added new revision row for Punch ID {punch_id}. Total rows now: {len(self.df)}")

        # Re-setup the vectorstore with the updated DataFrame
        # TODO: For better performance, consider implementing incremental updates to the vectorstore
        try:
            # Only update vectorstore with the new document instead of rebuilding everything
            if self.vectorstore:
                # Create a document from the new record
                content_cols = [col for col in self.df.columns if col in new_record.index]
                content_parts = [f"{col}: {new_record[col]}" for col in content_cols]
                content = "\n".join(content_parts)
                
                metadata = {"_original_index": len(self.df) - 1}
                # Add standard metadata fields if they exist
                metadata_map = {
                    "punch_id": 'Punch ID',
                    "task_number": 'Task Number',
                    "item": 'Item',
                    "item_type": 'ItemType',
                    "discipline": 'Disc',
                    "form_type": 'Form Type', 
                    "punch_status": 'Punch Status',
                    "punch_type": 'Type Of Punch'
                }
                
                for key, col_name in metadata_map.items():
                    if col_name in new_record:
                        metadata[key] = new_record[col_name]
                
                new_doc = Document(page_content=content, metadata=metadata)
                
                # Add just this document to the vectorstore
                self.vectorstore.add_documents([new_doc])
                logger.info("Added new document to vectorstore.")
            else:
                # If vectorstore doesn't exist, rebuild everything
                self.setup()
                logger.info("Vectorstore rebuilt after adding resolution.")
        except Exception as e:
            logger.error(f"Error updating vectorstore after adding resolution: {e}")
            traceback.print_exc()
            # Vectorstore might be in an inconsistent state now

        logger.info(f"Resolution added for Punch ID {punch_id}. New status: {new_status}. Revision: {revision}")
