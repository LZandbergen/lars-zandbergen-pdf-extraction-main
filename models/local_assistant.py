from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel, AutoModelForSeq2SeqLM
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from peft.peft_model import PeftModel
import torch
import pandas as pd
from pathlib import Path
import os
import shutil

import sys
sys.path.append('../pdf_parsing')
from pdf_parsing import pdf2html, scrape_html

# NOT USED AT THE MOMENT
class LocalLLMAssistant:
    def __init__(self, model_name: str):
        self.model_name = model_name

        # Load configurations for question encoder and generator
        #! PROBABLY WRONG TO USE SELF.MODEL_NAME HERE
        question_encoder_config = AutoConfig.from_pretrained(self.model_name)
        generator_config = AutoConfig.from_pretrained(self.model_name)
        
        # Initialize the tokenizer with question encoder and generator
        self.tokenizer = RagTokenizer.from_pretrained(
            self.model_name,
            question_encoder=question_encoder_config,
            generator=generator_config
        )
        # self.tokenizer = RagTokenizer.from_pretrained(self.model_name, question_encoder="facebook/dpr-question_encoder-single-nq-base", generator="facebook/bart-large")
        
        # Initialize the retriever
        self.retriever = RagRetriever.from_pretrained(
            self.model_name,
            index_name="custom",
            use_dummy_dataset=False
        )
        
        # Load the RAG model with sub-configurations
        self.model = RagTokenForGeneration.from_pretrained(
            self.model_name,
            question_encoder=question_encoder_config,
            generator=generator_config,
            retriever=self.retriever
        )
        # # self.model = RagTokenForGeneration.from_pretrained(self.model_name, question_encoder="facebook/dpr-question_encoder-single-nq-base", generator="facebook/bart-large", retriever=self.retriever)
        
        # Set the retriever to the model
        self.model.set_retriever(self.retriever)
        # Initialize the pipeline
        self.pipe = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=0)
        # Initialize the HuggingFace pipeline for LangChain
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.embeddings = HuggingFaceEmbeddings()
        self.retriever = None
        self.qa_chain = None

    def upload_file(self, filename: str):
        loader = PyMuPDFLoader(filename)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = vectorstore.as_retriever()

        # Initialize custom retriever with FAISS index
        self.retriever = RagRetriever.from_pretrained(
            self.model_name,
            index_name="custom",
            passages_path=None,
            index_path=None,
            use_dummy_dataset=False
        )
        # Update the model's retriever
        self.model.set_retriever(self.retriever)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff"
        )

    def predict(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("You need to upload a file first using upload_file.")
        return self.qa_chain.run(question)

class PDFProcessor:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.temp_dir = self.cache_dir / "temp"
        self.cache_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str) -> str:
        """Process PDF and return markdown content"""
        pdf_path = Path(pdf_path)
        
        # Create temp directory for processing
        temp_process_dir = self.temp_dir / pdf_path.stem
        temp_process_dir.mkdir(exist_ok=True)
        
        try:
            # Use pdf2html functions
            split_paths = pdf2html.split_files(str(pdf_path), str(temp_process_dir))
            html_paths = pdf2html.pdf2html(split_paths)
            
            # Convert HTML to DataFrame
            df = scrape_html.scrape_html(html_paths)
            
            # Convert DataFrame to markdown
            markdown_content = df.to_markdown(index=False)
            
            return markdown_content
        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_process_dir)

class LocalPDFAssistant:
    def __init__(self, model_name, tokenizer_path, cache_dir="cache"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
        # Update embeddings initialization with model_kwargs
        self.embeddings = LlamaCppEmbeddings(
            model_path=model_name,
        )
        self.vectorstore = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.markdown_dir = self.cache_dir / "markdown"
        self.markdown_dir.mkdir(exist_ok=True)
        self.vector_dir = self.cache_dir / "vectors"
        self.vector_dir.mkdir(exist_ok=True)
        self.pdf_processor = PDFProcessor(cache_dir)

    def _get_cache_path(self, pdf_path):
        pdf_hash = str(hash(str(Path(pdf_path).resolve())))
        return self.markdown_dir / f"{pdf_hash}.md", self.vector_dir / f"{pdf_hash}"

    def _df_to_markdown(self, df: pd.DataFrame):
        return df.to_markdown(index=False)

    def _process_pdf(self, pdf_path):
        temp_dir = self.cache_dir / "temp_pdfs"
        temp_dir.mkdir(exist_ok=True)
        
        split_paths = pdf2html.split_files(pdf_path, str(temp_dir))
        html_paths = pdf2html.pdf2html(split_paths)
        df = scrape_html.scrape_html(html_paths)
        
        # Cleanup temp files
        for path in split_paths + html_paths:
            os.remove(path)
            
        return df

    def upload_file(self, pdf_path):
        markdown_path, vector_path = self._get_cache_path(pdf_path)
        
        # Check if cached version exists
        if not markdown_path.exists():
            # Ensure the directory exists
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            df = self._process_pdf(pdf_path)
            markdown_content = self._df_to_markdown(df)
            markdown_path.write_text(markdown_content)
        
        # Load or create vector store
        if vector_path.exists():
            self.vectorstore = FAISS.load_local(str(vector_path), self.embeddings)
        else:
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            text = markdown_path.read_text()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.create_documents([text])
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            self.vectorstore.save_local(str(vector_path))

    def predict(self, question):
        if not self.vectorstore:
            raise ValueError("No PDF uploaded. Please upload a PDF file first.")
        docs = self.vectorstore.similarity_search(question)
        context = " ".join([doc.page_content for doc in docs])
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}"
            }
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after the last assistant message
        parts = answer.split("[/INST]")
        return parts[-1].strip() if len(parts) > 1 else answer.strip()

class FineTunedLlamaAssistant:
    def __init__(
        self,
        base_model_path: str = "models/meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = "models/fine-tuned/llama-3-8b-ave-amazon",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cuda",
            torch_dtype=torch.float16
        )
        
        # Load adapter and create PeftModel
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        # Create pipeline for generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            device_map="cuda"
        )
    
    def predict(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert in extracting product features from technical documents."
            },
            {
                "role": "user",
                "content": f"Please provide the attributes for the following product description:\n\n{prompt}"
            }
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        response = self.pipe(formatted_prompt)[0]['generated_text']
        
        # Extract response after the last assistant message
        parts = response.split("[/INST]")
        return parts[-1].strip() if len(parts) > 1 else response.strip()