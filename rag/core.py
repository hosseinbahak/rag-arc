# rag/core.py - Add/verify these prints
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
# Add this import for better error handling
import traceback

class EquipmentInspectionRAG:
    def __init__(self, excel_path: str):
        print(f"Initializing EquipmentInspectionRAG with Excel path: {excel_path}")
        self.excel_path = excel_path
        self.df = None
        self.vectorstore = None
        # Ensure OPENAI_API_KEY is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
             print("Error: OPENAI_API_KEY environment variable not set.")
             raise ValueError("OPENAI_API_KEY environment variable not set.")

        try:
            # Using gpt-4-turbo-preview for potentially better Farsi handling and performance
            self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, api_key=openai_api_key)
            print("ChatOpenAI model initialized.")
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {e}")
            self.llm = None # Set to None if initialization fails


        try:
            self.setup() # Initial data load and vectorstore creation
            print("Setup complete.")
        except Exception as e:
            print(f"Error during initial setup: {e}")
            traceback.print_exc() # Print full traceback
            self.df = pd.DataFrame() # Ensure df is a DataFrame even if setup fails
            self.vectorstore = None
            self.retriever = None

        self.graph = None # Graph will be created on first query

    def setup(self):
        """Loads data from Excel, prepares documents, and initializes vectorstore."""
        print(f"Attempting to load data from {self.excel_path}...")
        try:
            # Read, convert all to string to avoid type issues, fill NaN
            self.df = pd.read_excel(self.excel_path)
            print("Excel file read successfully.")
            # Important: Convert *all* columns to string and fillna AFTER reading
            # This prevents errors with mixed types or NaNs during string operations later
            for col in self.df.columns:
                 self.df[col] = self.df[col].astype(str).fillna("None")
            
            print(f"DataFrame loaded and cleaned. Found {len(self.df)} rows.")
            if len(self.df) == 0:
                 print("Warning: DataFrame is empty after loading.")


            print("Creating documents for vectorstore...")
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
                print(f"Warning: Missing expected columns in Excel for RAG content: {missing_cols}. These will be omitted.")

            # Ensure all metadata columns exist
            existing_metadata_map = {k: v for k, v in metadata_map.items() if v in self.df.columns}
            missing_metadata_cols = [v for k, v in metadata_map.items() if v not in self.df.columns]
            if missing_metadata_cols:
                 print(f"Warning: Missing expected columns in Excel for metadata: {missing_metadata_cols}. Metadata may be incomplete.")


            for index, row in self.df.iterrows():
                # Construct content string from existing columns
                content_parts = [f"{col}: {row[col]}" for col in existing_content_cols]
                content = "\n".join(content_parts)

                # Construct metadata dictionary from existing columns
                metadata = {"_original_index": index} # Always add original index

                for key, col_name in existing_metadata_map.items():
                     metadata[key] = row[col_name]

                documents.append(Document(page_content=content, metadata=metadata))

            print(f"Created {len(documents)} documents for vectorstore.")
            if len(documents) == 0 and len(self.df) > 0:
                 print("Error: No documents created despite having rows in DataFrame. Check column names used for content.")

            print("Initializing Chroma vectorstore...")
            if documents: # Only attempt to create vectorstore if there are documents
                # Using in-memory Chroma DB. Data is not persistent across runs.
                # For persistence, configure a directory: Chroma(persist_directory="./chroma_db", ...)
                # And call .persist() after adding documents.
                 try:
                    self.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=OpenAIEmbeddings() # Ensure OPENAI_API_KEY is set for this too
                    )
                    print("Vectorstore initialized.")
                    self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    print("Retriever created.")
                 except Exception as e:
                     print(f"Error initializing Chroma or Embeddings: {e}")
                     traceback.print_exc()
                     self.vectorstore = None
                     self.retriever = None

            else:
                 print("No documents created, skipping vectorstore initialization.")
                 self.vectorstore = None
                 self.retriever = None


        except FileNotFoundError:
            print(f"Error: Excel file not found at {self.excel_path}")
            self.df = pd.DataFrame() # Initialize empty DataFrame to prevent errors
            self.vectorstore = None
            self.retriever = None
        except Exception as e:
            print(f"An unexpected error occurred during setup: {e}")
            traceback.print_exc() # Print full traceback
            self.df = pd.DataFrame() # Initialize empty DataFrame
            self.vectorstore = None
            self.retriever = None


    def create_graph(self):
        """Defines and compiles the LangGraph workflow."""
        print("Attempting to create RAG graph...")
        if not self.llm or not self.retriever:
            print("Cannot create graph: LLM or Retriever not initialized.")
            self.graph = None
            return self.graph

        class State(Dict):
            query: str
            retrieved_documents: Optional[List[Document]] = None # Specify type hint
            context: Optional[str] = None
            answer: Optional[str] = None

        def retrieve(state: State) -> State:
            print(f"Graph node 'retrieve' called for query: {state['query'][:50]}...")
            if not self.retriever:
                 print("Retriever not initialized in retrieve node.")
                 state["retrieved_documents"] = []
                 return state

            try:
                docs = self.retriever.get_relevant_documents(state["query"])
                state["retrieved_documents"] = docs
                print(f"Retrieved {len(docs)} documents.")
                # print("Retrieved document contents (first 100 chars):")
                # for i, doc in enumerate(docs):
                #      print(f" Doc {i}: {doc.page_content[:100]}...")

            except Exception as e:
                print(f"Error during document retrieval: {e}")
                traceback.print_exc()
                state["retrieved_documents"] = []

            return state

        def generate_context(state: State) -> State:
            print("Graph node 'generate_context' called.")
            if not state["retrieved_documents"]:
                state["context"] = "No relevant documents found in the database."
                print("No documents to generate context from.")
            else:
                 context = "\n\n".join(doc.page_content for doc in state["retrieved_documents"])
                 state["context"] = context
                 # print(f"Context generated (first 200 chars): {context[:200]}...")

            return state

        def generate_answer(state: State) -> State:
            print("Graph node 'generate_answer' called.")
            if not self.llm:
                state["answer"] = "LLM not initialized. Cannot generate answer."
                print("LLM not initialized in generate_answer node.")
                return state

            if not state["context"] or state["context"] == "No relevant documents found in the database.":
                 state["answer"] = "Based on the available data, I cannot find relevant information to answer your question."
                 print("No context provided for answer generation.")
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
                print("Invoking LLM chain...")
                answer = chain.invoke({"context": state["context"], "query": state["query"]})
                state["answer"] = answer
                print("Answer generated successfully.")
            except Exception as e:
                print(f"Error during LLM chain invocation: {e}")
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
        print("RAG graph compiled.")
        return self.graph

    def query(self, question: str) -> str:
        """Runs a query through the RAG graph."""
        print(f"Received query: {question}")
        if not hasattr(self, 'graph') or self.graph is None:
            print("Graph not initialized or failed initialization, attempting to create.")
            self.create_graph() # Try creating it now if it failed before

        if not self.graph:
             print("RAG graph is not available.")
             return "RAG system is not fully initialized. Cannot process query."

        try:
            print("Invoking RAG graph...")
            result = self.graph.invoke({"query": question})
            print("Graph invocation complete.")
            return result.get("answer", "No answer generated by the graph.")
        except Exception as e:
             print(f"Error during graph invocation: {e}")
             traceback.print_exc()
             return "An error occurred while processing your query."


    def filter_punches(self, Disc: str = "", ItemType: str = "", Punch_Status: str = "") -> List[Dict[str, Any]]:
        """Filters the DataFrame based on exact matches for specified columns."""
        print(f"Filtering punches with Disc='{Disc}', ItemType='{ItemType}', Punch_Status='{Punch_Status}'")
        if self.df is None or self.df.empty:
             print("DataFrame is not loaded or is empty.")
             return [] # Return empty list if no data

        df = self.df.copy()
        original_count = len(df)

        # Use boolean indexing for exact matches where specified
        # Ensure columns exist AND handle potential None values from fillna appropriately
        # Using .astype(str) again defensively before comparison
        if Disc:
            if 'Disc' in df.columns:
                df = df[df['Disc'].astype(str).str.strip().str.lower() == Disc.strip().lower()]
                print(f"Filtered by Disc='{Disc}', remaining rows: {len(df)}")
            else:
                 print("Warning: 'Disc' column not found for filtering.")

        if ItemType:
            if 'ItemType' in df.columns:
                df = df[df['ItemType'].astype(str).str.strip().str.lower() == ItemType.strip().lower()]
                print(f"Filtered by ItemType='{ItemType}', remaining rows: {len(df)}")
            else:
                 print("Warning: 'ItemType' column not found for filtering.")

        if Punch_Status:
            if 'Punch Status' in df.columns:
                 # Ensure the status we are filtering by exists in the data
                 # And handle potential case sensitivity or extra spaces
                df = df[df['Punch Status'].astype(str).str.strip().str.lower() == Punch_Status.strip().lower()]
                print(f"Filtered by Punch Status='{Punch_Status}', remaining rows: {len(df)}")
            else:
                 print("Warning: 'Punch Status' column not found for filtering.")

        print(f"Finished filtering. Started with {original_count}, ended with {len(df)}.")
        # Convert DataFrame rows to list of dictionaries
        return df.to_dict(orient="records")

    def add_punch_resolution(self, punch_id: str, new_status: str = "1",
                             resolution_text: str = "", revision: str = "") -> None:
        """Adds a new row representing a resolution/revision for an existing punch."""
        print(f"Received request to add resolution for Punch ID: {punch_id}")
        if self.df is None or self.df.empty:
             print("DataFrame is not loaded or is empty.")
             raise ValueError("DataFrame is not loaded. Cannot add resolution.")

        # We assume 'Punch ID' is the identifier for the punch *thread*.
        # Find the latest revision for this punch ID thread.
        if 'Punch ID' not in self.df.columns or 'RevisionNumber' not in self.df.columns:
            raise ValueError("'Punch ID' or 'RevisionNumber' column missing in data. Cannot track resolutions.")

        # Find all rows belonging to the punch thread identified by `punch_id`
        # Use .astype(str).str.strip() for robust comparison
        punch_thread_rows = self.df[self.df['Punch ID'].astype(str).str.strip() == str(punch_id).strip()].copy()

        if punch_thread_rows.empty:
             print(f"Punch thread with ID '{punch_id}' not found in DataFrame.")
             raise ValueError(f"Punch ID {punch_id} not found to add resolution.")

        # Find the latest revision within this thread to base the new row on
        try:
            # Convert RevisionNumber to numeric, coercing errors to NaN
            punch_thread_rows['RevisionNumber_numeric'] = pd.to_numeric(punch_thread_rows['RevisionNumber'], errors='coerce')
            # Sort by numeric revision, placing NaNs (errors) first to ensure a valid number is picked if possible
            latest_revision_row = punch_thread_rows.sort_values(by='RevisionNumber_numeric', ascending=False, na_position='last').iloc[0]
            latest_revision_number = latest_revision_row.get('RevisionNumber_numeric') # Use .get to be safe
            if pd.isna(latest_revision_number):
                 latest_revision_number = -1 # Use a flag if not a valid number
                 print(f"Warning: Latest revision number for {punch_id} is not numeric ('{latest_revision_row.get('RevisionNumber', 'N/A')}'). Cannot auto-increment.")
            else:
                 latest_revision_number = int(latest_revision_number) # Convert to int if numeric


        except Exception as e:
            print(f"Warning: Error finding latest revision number for {punch_id}: {e}. Falling back to simple last row and revision '0'.")
            traceback.print_exc()
            # If numerical sort fails, just grab the last row found by ID
            latest_revision_row = punch_thread_rows.iloc[-1].copy()
            latest_revision_number = -1 # Indicate unknown/non-numeric revision


        # Create the new resolution record based on the latest revision row found
        new_record = latest_revision_row.copy()

        # Update fields for the new revision row
        # The new row *is* a revision of the punch thread identified by 'Punch ID'
        new_record['PrvPunchId'] = latest_revision_row['Punch ID'] # Link back to the ID of the previous revision row (which is the thread ID if only initial row exists)
        new_record['Punch ID'] = latest_revision_row['Punch ID'] # The new row keeps the same Punch ID (thread ID)
        new_record['Punch Status'] = str(new_status) # Ensure status is string
        new_record['Punch.1'] = resolution_text # Add resolution text (assuming 'Punch.1' is the resolution column)
        new_record['RevisionDate'] = revision # Use the provided revision identifier (timestamp/string)

        # Increment Revision Number
        if latest_revision_number != -1:
            new_record['RevisionNumber'] = str(latest_revision_number + 1)
            print(f"Incremented revision number for {punch_id} from {latest_revision_number} to {new_record['RevisionNumber']}.")
        else:
             # If previous was non-numeric or unknown, set a default or specific value
             # Let's use the provided revision + a suffix if the original was non-numeric
             original_rev_str = latest_revision_row.get('RevisionNumber', '0')
             new_record['RevisionNumber'] = f"{original_rev_str}_res_{revision}" # Example: "RevA_res_2023-10-27"
             print(f"Set revision number for {punch_id} based on original '{original_rev_str}' and new revision text '{revision}'.")


        # Remove the temporary numeric column
        if 'RevisionNumber_numeric' in new_record:
             del new_record['RevisionNumber_numeric']


        # Append the new row to the DataFrame
        # Need to ensure columns match exactly for concat - pandas handles this if columns are same
        # Using pd.DataFrame([new_record]) creates a 1-row DataFrame to concat
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)

        print(f"Added new revision row for Punch ID {punch_id}. Total rows now: {len(self.df)}")

        # Re-setup the vectorstore with the updated DataFrame
        # This is inefficient but necessary with the current in-memory Chroma and data structure
        # A persistent DB and incremental updates would be better
        try:
             self.setup() # Rebuild vectorstore with the new data
             print("Vectorstore rebuilt after adding resolution.")
        except Exception as e:
             print(f"Error rebuilding vectorstore after adding resolution: {e}")
             traceback.print_exc()
             # Vectorstore might be in an inconsistent state now


        print(f"Resolution added for Punch ID {punch_id}. New status: {new_status}. Revision: {revision}")

