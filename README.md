# LLM WITH RAG PROJECT
This project demonstrates the integration of a large language model (LLM) with a Retrieval-Augmented Generation (RAG) pipeline using various Python libraries. The project is set up in Google Colab, making it easy to run and experiment with.
## Table of content

- Installation
- Usage
- Conclusion
- Acknowledgement

## Installation
To get started, install the necessary Python libraries by running the following commands:
```python
!pip install -q transformers==4.41.2
!pip install -q bitsandbytes==0.43.1
!pip install -q accelerate==0.31.0
!pip install -q langchain==0.2.5
!pip install -q langchainhub==0.1.20
!pip install -q langchain-chroma==0.1.1
!pip install -q langchain-community==0.2.5
!pip install -q langchain_huggingface==0.0.3
!pip install -q python-dotenv==1.0.1
!pip install -q pypdf==4.2.0
!pip install -q numpy==1.24.4
``` 

## Usage

### Import Necessary Libraries
First, import all necessary libraries for the project:
```python
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
```

## Load PDF Document
Specify the path to your PDF document and load it using the PyPDFLoader:
```python
Loader = PyPDFLoader
FILE_PATH = "/content/YOLOv10_Tutorials.pdf"
loader = Loader(FILE_PATH)
documents = loader.load()
```

### Process and Split the Document
Use a text splitter to process the document and prepare it for embedding:
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
```

### Create and Configure the Model
Set up the LLM with the required configurations:
```python
bnb_config = BitsAndBytesConfig()
model = AutoModelForCausalLM.from_pretrained("gpt-neo-2.7B", config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("gpt-neo-2.7B")
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = HuggingFacePipeline(pipeline=pipeline)
```

### Set Up Retrieval-Augmented Generation
Create a retrieval chain to integrate the model with a retriever:
```python
retriever = Chroma(texts=texts, embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
chain = ConversationalRetrievalChain(llm=llm, retriever=retriever)
```

### Running the Model
Use the model to generate responses based on the retrieved documents:
```python
memory = ConversationBufferMemory()
history = ChatMessageHistory(memory=memory)
input_text = "Your query here"

response = chain.run(input_text=input_text, chat_history=history)
print(response)
```

## Conclusion
This project illustrates how to set up a Retrieval-Augmented Generation pipeline using a large language model. By following the steps above, you can customize and expand this setup for various applications, including document retrieval, question answering, and conversational agents.

## Acknowledgements

 - [Hugging Face](https://huggingface.co)
 - [LangChain](https://www.langchain.com)
 - [Google Colab](https://colab.research.google.com)
