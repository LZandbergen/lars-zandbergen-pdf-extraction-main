from openai import OpenAI
import os
import httpx
import json
import time
from typing import List, Optional
from pathlib import Path
import weave
from weave import Model
import base64
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

weave.init('openai')

class OpenAIAssistant:
    """
    A class to interact with the OpenAI API to create an assistant for answering questions based on a PDF file.

    Attributes:
        client (OpenAI): Client for interacting with OpenAI API.
        assistant_id (Optional[str]): ID of the created assistant. None until an assistant is created.
    """

    @weave.op(name='Init assistant')
    def __init__(self, model_name: str = "gpt-4o", mapping_file=(Path(__file__).parent / 'PDF_Assistant_mapping.json').resolve()) -> None:
        """
        Initializes the PDFAssistant with the API key from environment variables.
        
        Args:
            model_name (str): Name of the model to use (defaults to "gpt-4o")
            mapping_file (Path): Path to the mapping file
        """
            
        self.model_name = model_name
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API Key not found in environment variables")
        self.client = OpenAI(api_key=api_key, timeout=httpx.Timeout(45.0, read=15.0, write=30.0, connect=5.0))
        self.assistant_id: Optional[str] = None
        self.vector_store_id: Optional[str] = None
        self.message_file_id: Optional[str] = None
        self.mapping_file = mapping_file
        self.mappings = self.load_mappings()
        self.vector_store_file_ids: List[str] = []

    def get_cache_key(self, filename: str) -> str:
        """
        Generate a cache key for the assistant mapping that includes the model name.
        
        Args:
            filename (str): The path to the PDF file
            
        Returns:
            str: Cache key combining filename and model name
        """
        return f"{filename}_{self.model_name}"

    def load_mappings(self):
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                content = f.read().strip()
                if content:  # Check if file has content
                    return json.loads(content)
                return {}  # Return empty dict for empty file
        else:
            # Create the file with an empty JSON object
            with open(self.mapping_file, 'w') as f:
                json.dump({}, f)
            return {}
        
    def save_mappings(self):
        with open(self.mapping_file, 'w') as f:
            json.dump(self.mappings, f, indent=4)


    def check_files_exist(self) -> bool:
        # Checks if the assistant and vector store files are still stored
        try:
            self.client.beta.assistants.retrieve(self.assistant_id)
        except Exception:
            print(f"Assistant {self.assistant_id} not found in API.")
            return False

        # Check vector store files
        vector_store_files = self.client.beta.vector_stores.files.list(
            vector_store_id=self.vector_store_id
        ).data
        existing_file_ids = [file.id for file in vector_store_files]
        if not all(file_id in existing_file_ids for file_id in self.vector_store_file_ids):
            print("Some vector store files are missing.")
            return False

        return True

    @weave.op(name='Upload and store files')
    def upload_and_store_files(self, filename: str):
        # Uploads files and stores their IDs
        vector_store = self.client.beta.vector_stores.create(
            name=f"Product Documents for {os.path.basename(filename)} ({self.model_name})"
        )
        self.vector_store_id = vector_store.id

        # Upload file for vector store and capture file IDs
        with open(filename, "rb") as file:
            self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=self.vector_store_id,
                files=[file]
            )

        # Get the uploaded vector store file IDs
        vector_store_files = self.client.beta.vector_stores.files.list(
            vector_store_id=self.vector_store_id
        ).data
        self.vector_store_file_ids = [vs_file.id for vs_file in vector_store_files]

        # Upload file for assistant usage
        with open(filename, "rb") as file:
            uploaded_file = self.client.files.create(
                file=file,
                purpose="assistants"
            )
            self.message_file_id = uploaded_file.id
            self.assistant_file_id = uploaded_file.id

        # Create new assistant
        assistant = self.client.beta.assistants.create(
            name=f"Assistant for {os.path.basename(filename)}, Lars ({self.model_name})",
            instructions="""You are an assistant specialized in extracting product features from PDF files.
            You always return just the JSON with no additional description or context.
            Any list should be converted to a string with commas separating the items.""",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}},
            model=self.model_name,
            temperature=0
        )
        self.assistant_id = assistant.id

        # Update mappings
        cache_key = self.get_cache_key(filename)
        self.mappings[cache_key] = {
            "assistant_id": self.assistant_id,
            "vector_store_id": self.vector_store_id,
            "message_file_id": self.message_file_id,
            "vector_store_file_ids": self.vector_store_file_ids,
            "assistant_file_id": self.assistant_file_id
        }
        self.save_mappings()

        print(f"Created new assistant for '{filename}' with model {self.model_name}")

    def upload_file(self, filename: str) -> None:
        """
        Uploads a file to the OpenAI API and creates an assistant related to that file.

        Args:
            filename (str): The path to the file to be uploaded.
        """
        filename = os.path.abspath(filename)
        cache_key = self.get_cache_key(filename)
        
        if (cache_key in self.mappings):
            # Reuse existing assistant for this file and model combination
            mapping = self.mappings[cache_key]
            self.assistant_id = mapping['assistant_id']
            self.vector_store_id = mapping['vector_store_id']
            self.vector_store_file_ids = mapping['vector_store_file_ids']
            self.message_file_id = mapping['message_file_id']
            print(f"Reusing existing assistant for '{filename}' with model {self.model_name}")

            # Check if the files are still stored
            files_ok = self.check_files_exist()

            if not files_ok:
                print("Associated files are missing. Re-uploading files.")
                self.upload_and_store_files(filename)
        else:
            self.upload_and_store_files(filename)

    @weave.op(name='Get answers')
    def predict(self, question: str) -> List[str]:
        """
        Asks a question to the assistant and retrieves the answers.

        Args:
            question (str): The question to be asked to the assistant.

        Returns:
            List[str]: A list of answers from the assistant.

        Raises:
            ValueError: If the assistant has not been created yet.
        """
        if self.assistant_id is None:
            raise ValueError(
                "Assistant not created. Please upload a file first.")

        thread = self.client.beta.threads.create(
            messages=[{"role": "user",
                       "content": question,
                       "attachments": [{"file_id": self.message_file_id, "tools": [{"type": "file_search"}]}]
                       }]
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id
        )

        while run.status != 'completed':
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id)
            time.sleep(0.1)

        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id, run_id=run.id)

        return [message.content[0].text.value for message in messages.data if message.role == "assistant"]

class ClaudeAssistant(Model):
    """
    A class to interact with the Anthropic API to create an assistant for answering questions based on a PDF file.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022") -> None:
        """
        Initializes the ClaudeAssistant with the API key from environment variables.
        """
        self.model_name = model_name
        api_key: Optional[str] = os.getenv("CLAUDE_API_KEY")
        if api_key is None:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key, default_headers={"antropic-beta": "pdfs-2024-09-25"})
        self.pdf_data = None

    def upload_file(self, filename: str) -> None:
        """
        Encodes the PDF file in base64 for use with Claude's API.

        Args:
            filename (str): Path to the PDF file
        """
        filename = os.path.abspath(filename)
        with open(filename, "rb") as pdf_file:
            binary_data = pdf_file.read()
            self.pdf_data = base64.standard_b64encode(binary_data).decode('utf-8')

    @weave.op(name='Get answers')
    def predict(self, question: str) -> List[str]:
        """
        Sends a question along with the PDF content to the Claude model and retrieves the answer.

        Args:
            question (str): The question to be asked.

        Returns:
            List[str]: A list containing the assistant's answer.
        """
        if self.pdf_data is None:
            raise ValueError("No PDF file uploaded. Please upload a file first.")

        content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": self.pdf_data
                },
                "cache_control": {"type": "ephemeral"}
            },
            {"type": "text", "text": question}
        ]

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": content}]
            )
            return [response.content[0].text]
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            return [f"Error: {str(e)}"]
