# rag/core.py - Modified to use Ollama models
from typing import Dict, List, Optional, Any
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 

# from langchain_ollama import Ollama
from langgraph.graph import END, StateGraph
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import traceback
import logging
from datetime import datetime

# For handling SelfQueryRetriever functionality
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EquipmentInspectionRAG")

class EquipmentInspectionRAG:
    def __init__(self, excel_path: str, default_retrieval_k: Optional[int] = None, 
                 llm_model: str = "gemma2:27b", embedding_model: str = "nomic-embed-text"):
        """
        Initialize the RAG system for equipment inspection data.

        Args:
            excel_path: Path to the Excel file containing punch data
            default_retrieval_k: Max number of documents to retrieve. 
                                 If None, defaults to total number of rows in Excel or 1000.
            llm_model: Name of the Ollama model to use for LLM operations
            embedding_model: Name of the Ollama model to use for embeddings
        """
        logger.info(f"Initializing EquipmentInspectionRAG with Excel path: {excel_path}")
        self.excel_path = excel_path
        self.df = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.default_retrieval_k = default_retrieval_k
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        try:
            # Initialize Ollama LLM
            self.llm = Ollama(model=self.llm_model, temperature=0)
            logger.info(f"Ollama model '{self.llm_model}' initialized.")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            # self.llm remains None

        try:
            self.setup()
            logger.info("Setup complete.")
        except Exception as e:
            logger.error(f"Error during initial setup: {e}")
            traceback.print_exc()
            self.df = pd.DataFrame()
            self.vectorstore = None
            self.retriever = None

        self.graph = None

    def setup(self):
        logger.info(f"Attempting to load data from {self.excel_path}...")
        try:
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"Excel file read successfully. Found {len(self.df)} rows with columns: {list(self.df.columns)}")
            
            for col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().fillna("None")
            
            logger.info(f"DataFrame loaded and cleaned. Found {len(self.df)} rows.")
            if len(self.df) == 0:
                logger.warning("DataFrame is empty after loading. RAG system may not function correctly.")
            
            # Determine MAX_DOCS_TO_RETRIEVE
            if self.default_retrieval_k is not None:
                MAX_DOCS_TO_RETRIEVE = self.default_retrieval_k
            elif self.df is not None and not self.df.empty:
                MAX_DOCS_TO_RETRIEVE = len(self.df) # Max possible docs
            else:
                MAX_DOCS_TO_RETRIEVE = 1000 # A reasonable fallback
            logger.info(f"Retriever will be configured to fetch up to {MAX_DOCS_TO_RETRIEVE} documents.")

            logger.info("Creating documents for vectorstore...")
            documents = []

            # Define columns to include in the RAG document content and metadata
            # KEYS here are what SelfQueryRetriever will use. VALUES are your Excel column names.
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
            
            # These columns will form the main text content of each document in the vector store.
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

            existing_metadata_map = {k: v for k, v in metadata_map.items() if v in self.df.columns}
            missing_metadata_cols = [v for k, v in metadata_map.items() if v not in self.df.columns]
            if missing_metadata_cols:
                logger.warning(f"Missing expected columns in Excel for metadata: {missing_metadata_cols}. Metadata may be incomplete.")

            if not self.df.empty:
                for index, row in self.df.iterrows():
                    content_parts = [f"{col.replace(' ', '_')}: {row[col]}" for col in existing_content_cols]
                    content = "\n".join(content_parts)

                    metadata = {"_original_index": index}
                    for key, col_name in existing_metadata_map.items():
                        metadata[key] = row[col_name]

                    documents.append(Document(page_content=content, metadata=metadata))

            logger.info(f"Created {len(documents)} documents for vectorstore.")
            
            if not documents:
                logger.warning("No documents created. Vectorstore will be empty. RAG queries might not work as expected.")
                try:
                    # Initialize with OllamaEmbeddings instead of OpenAIEmbeddings
                    self.vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=self.embedding_model))
                    logger.info(f"Initialized an empty Chroma vectorstore with Ollama embeddings ({self.embedding_model}).")
                except Exception as e_chroma_empty:
                    logger.error(f"Could not initialize empty Chroma vectorstore: {e_chroma_empty}")
                    self.vectorstore = None
                self.retriever = None
                return

            logger.info(f"Initializing Chroma vectorstore with Ollama embeddings ({self.embedding_model})...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=OllamaEmbeddings(model=self.embedding_model)
            )
            logger.info("Vectorstore initialized.")

            metadata_field_info = [
                AttributeInfo(name="punch_id", description="The unique identifier of the punch", type="string"),
                AttributeInfo(name="task_number", description="The task number associated with the punch", type="string"),
                AttributeInfo(name="item", description="The item related to the punch", type="string"),
                AttributeInfo(name="item_type", description="The type of the item (e.g., 'EQUIPMENT', 'PIPING')", type="string"),
                AttributeInfo(name="discipline", description="The discipline (e.g., 'MECH', 'ELEC', 'INST')", type="string"),
                AttributeInfo(name="form_type", description="The type of form used", type="string"),
                AttributeInfo(name="punch_status", description="The status of the punch. Example values: '0' (open), '1' (closed), 'None'. This is a string.", type="string"),
                AttributeInfo(name="punch_type", description="The type of punch (e.g., 'A', 'B', 'C')", type="string"),
            ]
            document_description = "A record detailing an equipment inspection punch item, including its ID, description, status, discipline, and other attributes."

            if not self.llm:
                logger.error("LLM not initialized, cannot create SelfQueryRetriever. Falling back to simple retriever.")
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                logger.warning(f"Fell back to simple retriever with k=5 due to LLM init issues for SelfQuery.")
            else:
                try:
                    self.retriever = SelfQueryRetriever.from_llm(
                        llm=self.llm,
                        vectorstore=self.vectorstore,
                        document_contents=document_description,
                        metadata_field_info=metadata_field_info,
                        verbose=True,
                        search_kwargs={"k": MAX_DOCS_TO_RETRIEVE}
                    )
                    logger.info(f"SelfQueryRetriever created. Will retrieve up to {MAX_DOCS_TO_RETRIEVE} documents matching filters.")
                except Exception as e_sq:
                    logger.error(f"Error creating SelfQueryRetriever: {e_sq}. Falling back to simple retriever.")
                    traceback.print_exc()
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        except FileNotFoundError:
            logger.error(f"Excel file not found at {self.excel_path}")
            self.df = pd.DataFrame()
            self.vectorstore = None
            self.retriever = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during setup: {e}")
            traceback.print_exc()
            self.df = pd.DataFrame()
            self.vectorstore = None
            self.retriever = None

    def create_graph(self):
        logger.info("Attempting to create RAG graph...")
        if not self.llm or not self.retriever:
            logger.error("Cannot create graph: LLM or Retriever not initialized.")
            self.graph = None
            return self.graph

        class State(Dict): # type: ignore
            query: str
            retrieved_documents: Optional[List[Document]] = None
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
                state["context"] = "No relevant documents found in the database that match the query criteria."
                logger.warning("No documents to generate context from.")
            else:
                context_parts = []
                for i, doc in enumerate(state["retrieved_documents"]):
                    context_parts.append(f"--- DOCUMENT {i+1} ---\n{doc.page_content}\n")
                context = "\n".join(context_parts)
                state["context"] = context
                logger.info(f"Generated context from {len(state['retrieved_documents'])} documents. Context length: {len(context)} chars.")
            return state

        def generate_answer(state: State) -> State:
            logger.info("Graph node 'generate_answer' called.")
            if not self.llm:
                state["answer"] = "LLM not initialized. Cannot generate answer."
                logger.error("LLM not initialized in generate_answer node.")
                return state

            if not state["context"] or state["context"].startswith("No relevant documents found"):
                state["answer"] = "Based on the available data and applied filters, I cannot find relevant information to answer your question."
                logger.warning("No context provided or no documents found for answer generation.")
                return state

            num_retrieved_docs = len(state.get("retrieved_documents", []))
            
            max_possible_retrieved = self.default_retrieval_k
            if max_possible_retrieved is None:
                 max_possible_retrieved = len(self.df) if self.df is not None and not self.df.empty else 1000

            exhaustive_note = ""
            if num_retrieved_docs > 0:
                if num_retrieved_docs < max_possible_retrieved:
                    exhaustive_note = (
                        f"The following answer is based on {num_retrieved_docs} document(s) that best matched your query from the filtered set. "
                        f"More documents might exist that match the filter criteria but were not included due to relevance ranking or retrieval limits (limit was {max_possible_retrieved})."
                    )
                else:
                    if self.default_retrieval_k is None and (self.df is not None and not self.df.empty and max_possible_retrieved == len(self.df)):
                         exhaustive_note = f"The following answer is based on all {num_retrieved_docs} document(s) found in the database that matched your query criteria."
                    else:
                         exhaustive_note = f"The following answer is based on up to {num_retrieved_docs} document(s) that matched your query criteria (retrieval limit was {max_possible_retrieved})."

            template = f"""
            You are an expert assistant for an equipment inspection and punch management system.
            Use the following information from the database context to answer the user's question accurately and concisely.
            {exhaustive_note}

            Context:
            {{context}}

            Question: {{query}}
            System Overview:
            - The system manages equipment (`Item`s), their types (`ItemType`), disciplines (`Disc`), checklists (`Form Type`), task execution (`Task`), and defect tracking (`Punch`).
            - Data model is relational and normalized, enabling traceability, consistency, and integration with ERP or CMMS systems.
            - Each `ItemType` has one or more predefined checklists. When an `Item` is created, a `Task` is generated per checklist.
            - Punches are recorded when defects are identified and are tracked with status, type, and revision.
            - Resolution of a punch creates a new punch record referencing the original via `PrvPunchId`, enabling full revision tracking.

            ✅ Prompt System Instruction: Industrial Item & Punch Management System Context
            🔹 System Overview

            This is an industrial equipment management system that organizes and tracks the lifecycle of items (equipment), their associated checklists, task execution, and issue handling (Punches), across multiple disciplines.
            🔹 Entity Hierarchy and Relationships

                Disc (Discipline):

                    A top-level classification representing engineering domains such as:

                        EL → Electrical

                        Ins → Instrumentation

                ItemType:

                    Represents a category of equipment within a Discipline.

                    Example: "Junction Box" under Electrical.

                Item:

                    A specific, physical equipment unit.

                    Each Item belongs to a single ItemType.

            🔹 Checklist and Task Logic

                For each ItemType, one or more Checklists are defined to guide installation or quality assurance procedures.

                Each Checklist is labeled using a Form Type.

                When an Item is registered in the system, a Task is generated for each Checklist applicable to its ItemType.

                Each Task receives a unique identifier called a Task Number.

            🔹 Punch (Defect) Handling Process

                When an Item is installed in the field or factory, its checklists are completed.

                If an issue or defect is identified, a Punch is logged in the system.

                Each Punch has a Type of Punch, indicating its criticality:

                    Blocks pre-commissioning

                    Blocks commissioning

                    Blocks handover

                By default, all Punches are recorded with Punch Status = 0 (Initial Registration).

            🔹 Punch Resolution and Revision Tracking

                When a Punch is addressed:

                    A new record for that Punch is created.

                    The Punch Status is updated to 1 (Resolved).

                    The field PrvPunchId is set to the original Punch ID, linking the resolution to its source.

                    A Revision Number and Date of Resolution are also stored in the new record.

                This structure enables full auditability and traceability of defect lifecycles.

            🔹 System Goals

                Accurate tracking of equipment readiness through task completion.

                Transparent and categorized handling of defects.

                Full revision control and history tracking for Punch lifecycle management.


            Instructions:
            - Answer the question based ONLY on the provided context.
            - If the context does not contain the information to answer the question, respond with "Based on the available data, I cannot answer your question." or similar.
            - Do NOT make up information.
            - Answer in the same language as the question (English or Persian/Farsi).
            - If the user asks for statistics (counts, number of items, etc.):
                - Count the items *explicitly from the provided context*.
                - State the count you found in the context.
            - If the query involves filtering by status or other attributes, make sure to verify the exact values in the retrieved documents.
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            try:
                logger.info(f"Invoking Ollama LLM chain with {num_retrieved_docs} documents in context...")
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
        logger.info(f"Received query: {question}")
        if not hasattr(self, 'graph') or self.graph is None:
            logger.info("Graph not initialized or failed initialization, attempting to create.")
            self.create_graph()

        if not self.graph:
            logger.error("RAG graph is not available.")
            return "RAG system is not fully initialized. Cannot process query."
        
        if not self.retriever:
            logger.error("Retriever is not available (e.g. empty data source). Cannot process query.")
            return "RAG system's retriever is not initialized, possibly due to an empty data source. Cannot process query."

        try:
            logger.info("Invoking RAG graph...")
            initial_state = {"query": question}
            result = self.graph.invoke(initial_state)
            logger.info("Graph invocation complete.")
            return result.get("answer", "No answer generated by the graph.")
        except Exception as e:
            logger.error(f"Error during graph invocation: {e}")
            traceback.print_exc()
            return "An error occurred while processing your query."

    # Filter punches method
    def filter_punches(self, disc: str = "", item_type: str = "", punch_status: str = "") -> List[Dict[str, Any]]:
        logger.info(f"Filtering punches with disc='{disc}', item_type='{item_type}', punch_status='{punch_status}'")
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is not loaded or is empty.")
            return []

        df_filtered = self.df.copy() 
        original_count = len(df_filtered)

        if 'Punch Status' in df_filtered.columns and punch_status:
            unique_statuses = df_filtered['Punch Status'].unique()
            logger.debug(f"Unique Punch Status values in data for filtering: {unique_statuses}")

        if disc:
            if 'Disc' in df_filtered.columns:
                normalized_disc = disc.strip().lower()
                df_filtered = df_filtered[df_filtered['Disc'].str.lower().str.strip() == normalized_disc]
                logger.info(f"Filtered by Disc='{disc}', remaining rows: {len(df_filtered)}")
            else:
                logger.warning("Warning: 'Disc' column not found for filtering.")

        if item_type:
            if 'ItemType' in df_filtered.columns:
                normalized_item_type = item_type.strip().lower()
                df_filtered = df_filtered[df_filtered['ItemType'].str.lower().str.strip() == normalized_item_type]
                logger.info(f"Filtered by ItemType='{item_type}', remaining rows: {len(df_filtered)}")
            else:
                logger.warning("Warning: 'ItemType' column not found for filtering.")

        if punch_status:
            if 'Punch Status' in df_filtered.columns:
                normalized_status = punch_status.strip()
                logger.info(f"Looking for punch_status='{normalized_status}' (type: {type(normalized_status)})")
                
                df_filtered = df_filtered[df_filtered['Punch Status'].str.strip() == normalized_status]
                
                logger.info(f"Filtered by Punch Status='{punch_status}', remaining rows: {len(df_filtered)}")
                if not df_filtered.empty:
                    logger.debug(f"Sample of filtered data by status: {df_filtered['Punch Status'].head(5).tolist()}")
                else:
                    if not self.df[self.df['Punch Status'].str.strip() == normalized_status].empty:
                         logger.warning(f"No rows found with Punch Status='{punch_status}' after other filters, but status exists in original DF.")
                    else:
                         logger.warning(f"No rows found with Punch Status='{punch_status}' in the current filtered set. Original DF status values: {self.df['Punch Status'].unique()[:10]}")

            else:
                logger.warning("Warning: 'Punch Status' column not found for filtering.")

        logger.info(f"Finished filtering. Started with {original_count}, ended with {len(df_filtered)}.")
        return df_filtered.to_dict(orient="records")

    # Add punch resolution method
    def add_punch_resolution(self, punch_id: str, new_status: str = "1",
                             resolution_text: str = "", revision: str = "") -> None:
        logger.info(f"Received request to add resolution for Punch ID: {punch_id}")
        if self.df is None or self.df.empty:
            logger.error("DataFrame is not loaded or is empty. Cannot add resolution.")
            if self.df is None:
                raise ValueError("DataFrame is not initialized. Cannot add resolution.")

        if self.df.empty:
            logger.error(f"DataFrame is empty. Cannot find Punch ID {punch_id} to add resolution.")
            raise ValueError(f"DataFrame is empty. Punch ID {punch_id} not found.")

        if 'Punch ID' not in self.df.columns or 'RevisionNumber' not in self.df.columns:
            logger.error("'Punch ID' or 'RevisionNumber' column missing in data.")
            raise ValueError("'Punch ID' or 'RevisionNumber' column missing in data. Cannot track resolutions.")

        punch_thread_rows = self.df[self.df['Punch ID'].str.strip() == str(punch_id).strip()].copy()

        if punch_thread_rows.empty:
            logger.error(f"Punch thread with ID '{punch_id}' not found in DataFrame.")
            raise ValueError(f"Punch ID {punch_id} not found to add resolution.")

        try:
            punch_thread_rows['RevisionNumber_numeric'] = pd.to_numeric(punch_thread_rows['RevisionNumber'], errors='coerce')
            latest_revision_row_df = punch_thread_rows.sort_values(by='RevisionNumber_numeric', ascending=False, na_position='first')
            if latest_revision_row_df.empty: 
                logger.error(f"Could not determine latest revision for {punch_id} after attempting numeric sort.")
                raise ValueError(f"Error processing revisions for {punch_id}")
            latest_revision_row = latest_revision_row_df.iloc[0] 
            latest_revision_number = latest_revision_row.get('RevisionNumber_numeric')
            if pd.isna(latest_revision_number):
                latest_revision_number = -1 
                logger.warning(f"Latest revision number for {punch_id} is not numeric ('{latest_revision_row.get('RevisionNumber', 'N/A')}').")
            else:
                latest_revision_number = int(latest_revision_number)

        except Exception as e:
            logger.warning(f"Error finding latest revision number for {punch_id}: {e}. Falling back to simple last row and revision '0'.")
            traceback.print_exc()
            latest_revision_row = punch_thread_rows.iloc[-1].copy() 
            latest_revision_number = -1

        new_record_dict = latest_revision_row.to_dict() 

        if not revision:
            revision = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_record_dict['PrvPunchId'] = latest_revision_row.get('Punch ID', str(punch_id)) 
        new_record_dict['Punch ID'] = latest_revision_row.get('Punch ID', str(punch_id)) 
        new_record_dict['Punch Status'] = str(new_status).strip()
        
        if 'Punch' in new_record_dict:
            new_record_dict['Punch'] = resolution_text
        else:
            logger.warning("'Punch' column not found for resolution text.")
        
        new_record_dict['RevisionDate'] = revision

        if latest_revision_number != -1:
            new_record_dict['RevisionNumber'] = str(latest_revision_number + 1)
            logger.info(f"Incremented revision number for {punch_id} from {latest_revision_number} to {new_record_dict['RevisionNumber']}.")
        else:
            original_rev_str = latest_revision_row.get('RevisionNumber', '0')
            new_record_dict['RevisionNumber'] = f"{original_rev_str}_res_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Set complex revision number for {punch_id} based on original '{original_rev_str}'.")

        if 'RevisionNumber_numeric' in new_record_dict:
            del new_record_dict['RevisionNumber_numeric']

        new_record_df = pd.DataFrame([new_record_dict])
        self.df = pd.concat([self.df, new_record_df], ignore_index=True)


        logger.info(f"Added new revision row for Punch ID {punch_id}. Total rows now: {len(self.df)}")

        try:
            if self.vectorstore:
                if not hasattr(self.vectorstore, 'add_documents'):
                    logger.warning("Vectorstore does not support adding documents. Rebuilding entire vectorstore.")
                    self.setup()
                    return

                content_cols_for_new_doc = [col for col in self.df.columns if col in new_record_dict] 
                content_parts = [f"{col}: {new_record_dict[col]}" for col in content_cols_for_new_doc]
                content = "\n".join(content_parts)
                
                metadata = {"_original_index": len(self.df) - 1}
                metadata_map_for_new_doc = {
                    "punch_id": 'Punch ID', "task_number": 'Task Number', "item": 'Item',
                    "item_type": 'ItemType', "discipline": 'Disc', "form_type": 'Form Type', 
                    "punch_status": 'Punch Status', "punch_type": 'Type Of Punch'
                }
                for key, col_name in metadata_map_for_new_doc.items():
                    if col_name in new_record_dict: 
                        metadata[key] = new_record_dict[col_name]
                
                new_doc = Document(page_content=content, metadata=metadata)
                self.vectorstore.add_documents([new_doc])
                logger.info("Added new document to vectorstore.")
            else: 
                logger.warning("Vectorstore not available, attempting to rebuild with new data.")
                self.setup() 
        except Exception as e:
            logger.error(f"Error updating vectorstore after adding resolution: {e}")
            traceback.print_exc()
            logger.info("Attempting to rebuild vectorstore as a fallback.")
            self.setup()

        logger.info(f"Resolution added for Punch ID {punch_id}. New status: {new_status}. Revision: {revision}")

    def get_punch_history(self, punch_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving history for Punch ID: {punch_id}")
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is not loaded or is empty.")
            return []
            
        if 'Punch ID' not in self.df.columns:
            logger.error("'Punch ID' column missing in data.")
            return []
            
        normalized_id = str(punch_id).strip()
        punch_history_df = self.df[self.df['Punch ID'].str.strip() == normalized_id].copy()
        
        if punch_history_df.empty:
            logger.warning(f"No history found for Punch ID: {punch_id}")
            return []
            
        try:
            if 'RevisionNumber' in punch_history_df.columns:
                punch_history_df['RevisionNumber_numeric'] = pd.to_numeric(punch_history_df['RevisionNumber'], errors='coerce')
                # Sort also by RevisionDate as a secondary key if RevisionNumbers are not unique or non-numeric
                sort_by_cols = ['RevisionNumber_numeric']
                if 'RevisionDate' in punch_history_df.columns:
                    punch_history_df['RevisionDate'] = pd.to_datetime(punch_history_df['RevisionDate'], errors='coerce')
                    sort_by_cols.append('RevisionDate')
                
                punch_history_df = punch_history_df.sort_values(by=sort_by_cols, ascending=[True]*len(sort_by_cols), na_position='first')
                punch_history_df = punch_history_df.drop(columns=['RevisionNumber_numeric'])
            elif 'RevisionDate' in punch_history_df.columns:
                 punch_history_df['RevisionDate'] = pd.to_datetime(punch_history_df['RevisionDate'], errors='coerce')
                 punch_history_df = punch_history_df.sort_values(by='RevisionDate', ascending=True, na_position='first')
        except Exception as e:
            logger.warning(f"Unable to sort punch history by revision number/date: {e}")
        
        logger.info(f"Retrieved {len(punch_history_df)} history records for Punch ID {punch_id}")
        return punch_history_df.to_dict(orient="records")

    def get_summary_statistics(self) -> Dict[str, Any]:
        logger.info("Generating summary statistics")
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is not loaded or is empty.")
            return {"error": "No data available"}
            
        stats = {}
        stats["total_punches_entries"] = len(self.df) 
        
        if 'Punch ID' in self.df.columns:
            stats["total_unique_punch_threads"] = self.df['Punch ID'].nunique()

        # For more accurate "current" status, one would typically look at the latest revision of each punch.
        # This is a simplified version.
        if 'Punch Status' in self.df.columns:
            status_counts = self.df['Punch Status'].value_counts().to_dict()
            stats["status_distribution_all_entries"] = status_counts
            
        if 'Disc' in self.df.columns:
            disc_counts = self.df['Disc'].value_counts().to_dict()
            stats["discipline_distribution_all_entries"] = disc_counts
            
        if 'ItemType' in self.df.columns:
            item_type_counts = self.df['ItemType'].value_counts().to_dict()
            stats["item_type_distribution_all_entries"] = item_type_counts
            
        logger.info(f"Generated summary statistics: {stats}")
        return stats