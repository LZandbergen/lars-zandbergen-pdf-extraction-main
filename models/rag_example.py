from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os, os.path as op
import sys
from FlagEmbedding import BGEM3FlagModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import multiprocessing

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
from pdf_parsing import scrape_html, pdf2html

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Define paths to the base model and the fine-tuned adapter
    base_model_dir = "models/meta-llama/Llama-3.1-8B-Instruct"
    adapter_dir = "models/fine-tuned/llama-3-8b-ave-amazon"

    # Step 0: Parse the pdf file into markdown.
    pdf_path = "data/allshoes_catalogus/pdfs/allshoes_catalogus.pdf"
    temp_dir = "../../tmp/allshoes_catalogus/pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    # The markdown file will be stored in the same directory as the PDF
    pdf_dir = op.dirname(pdf_path)
    pdf_name = op.splitext(op.basename(pdf_path))[0]
    markdown_path = op.join(pdf_dir, f"{pdf_name}.md")

    if not op.exists(markdown_path):
        print("\nProcessing PDF to Markdown...")
        # Split the PDF into pages and convert each page to HTML, then scrape to markdown
        split_paths = pdf2html.split_files(pdf_path, temp_dir)
        html_paths = pdf2html.pdf2html(split_paths)
        df = scrape_html.scrape_html(html_paths)
        markdown_content = df.to_markdown(index=False)

        # Save the Markdown to a file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Markdown saved at: {markdown_path}")
    else:
        print(f"\nMarkdown file already exists at {markdown_path}, skipping re-processing.")

    # Step 1: Load the tokenizer from the base model directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        tokenizer.pad_token = tokenizer.eos_token  # Some LLAMA models require an explicit pad token
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # Step 2: Load the base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency if using a GPU
            device_map="auto"           # Automatically place model on available devices
        )
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        raise

    # Step 3: Load and apply the fine-tuned adapter
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            torch_dtype=torch.float16,  # Ensure dtype matches the base model
            device_map="auto"           # Automatically place adapter on available devices
        )
        model.eval()  # Set model to evaluation mode
        print("Fine-tuned adapter loaded and applied successfully.")
    except Exception as e:
        print(f"Error loading fine-tuned adapter: {e}")
        raise

    # Step 4: Create the text-generation pipeline
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,    # Adjust based on your needs
            temperature=0.7,       # Adjust for creativity vs. determinism
            top_p=0.9,             # Nucleus sampling
            device_map="auto"      # Ensure pipeline uses the correct device
        )
        print("Pipeline created successfully.")
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise

    # Step 5: Convert the markdown to a vectorized format

    # Vector store directory
    vectorstore_dir = op.join(pdf_dir, "vectorstore")
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore_path = op.join(vectorstore_dir, pdf_name)

    if not op.exists(vectorstore_path + ".faiss"):
        print("\nCreating vector store...")

        # Step 1: Read the markdown file
        with open(markdown_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Step 2: Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([text])  # docs is a list of Document objects

        # Step 3: Embed using the bge-m3 model
        # We'll wrap BGEM3FlagModel in a class that conforms to LangChain's Embeddings interface
        class BGEEmbeddings(Embeddings):
            def __init__(self, model_name='BAAI/bge-m3', use_fp16=True):
                self.bge_model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
            
            def embed_documents(self, texts):
                # texts is a list of strings
                result = self.bge_model.encode(texts, batch_size=12, max_length=8192)
                return result['dense_vecs']
            
            def embed_query(self, query):
                result = self.bge_model.encode([query], batch_size=1, max_length=8192)
                return result['dense_vecs'][0]

        embeddings = BGEEmbeddings()

        # Extract just the texts from docs
        texts = [doc.page_content for doc in docs]

        # Create the vector store
        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(vectorstore_path)

        print(f"Vector store created and saved at: {vectorstore_path}.faiss")

        # Step 5: Clear CUDA cache (if running on GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
    else:
        print("\nVector store already exists, skipping embedding.")

    # Step 6: Test the pipeline with a sample prompt
    prompt = (
        "Please provide the attributes for the following product description:\n\n"
        "6318962387"
    )

    try:
        output = pipe(prompt)
        generated_text = output[0]['generated_text']
        # Remove the original prompt from the response
        answer = generated_text[len(prompt):].strip()
        print("\nGenerated Answer:")
        print(answer)
    except Exception as e:
        print(f"Error during text generation: {e}")
        raise