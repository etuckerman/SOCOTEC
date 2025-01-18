This project involves the development and fine-tuning of a Large Language Model (LLM) to generate and execute function calls based on user prompts. The focus is on implementing a robust, scalable RAG (Retrieval-Augmented Generation) pipeline capable of answering questions from a document corpus with high accuracy.

The project began with basic PDF parsing but was later upgraded to use LlamaParse with advanced prompting to significantly enhance parsing accuracy. This shift improved the model’s ability to extract and understand complex content from documents, ensuring that important information is retained and accurately represented.  
Embedding Model Transition

Initially, the project employed smaller embedding models like "all-MiniLM-L6-v2" for document retrieval and query processing. However, after browsing the MTEB leaderboard on Hugging Face to explore state-of-the-art (SOTA) embedding models specifically trained with QWEN, larger and more advanced models were adopted. Notably, the BGE-M3 model, trained with QWEN, was incorporated, delivering better accuracy and retrieval results, significantly enhancing the overall performance of the RAG pipeline.  
RAG Pipeline Overhaul

The RAG pipeline underwent a significant overhaul, where a shift was made to incorporate larger models, including the QWEN-based BGE-M3 embeddings. This change allowed for improved query-document matching and better retrieval of relevant data. The pipeline now uses a summary approach, where chunks are processed using the QWEN model, rather than retrieving raw data directly from documents.  
Metrics and Performance Monitoring

Cosine similarity metrics were integrated into the pipeline to evaluate the relevance of retrieved data to the user’s query. An average similarity score was introduced, and a graph was included to visualize the performance of the pipeline over different queries. The similarity score for one of the initial queries reached approximately 81%, showcasing the model’s progress in terms of accuracy.  
Gradio Interface and User Interaction

A Gradio interface was implemented to allow for interactive querying and displaying RAG outputs. Adjustments were made to format the results for clarity and to enhance user experience. This interface provides users with the ability to easily query the document corpus and receive relevant data, summarized where necessary.  
Challenges and Solutions

One of the significant challenges faced was memory limitations when trying to use the Stella model for embeddings. This issue was encountered on the GPU plan being used, but it was mitigated by switching to a more efficient model.  
Additionally, the chunking process for document retrieval was adjusted to ensure that no critical sections of the document were missed, especially when summarizing large content sections.

The project has successfully implemented a high-performance RAG pipeline capable of answering user queries from a document corpus. Through the use of advanced embedding models and optimization techniques, the accuracy and efficiency of document retrieval and question answering have been significantly enhanced. The interactive Gradio interface provides an intuitive means for users to engage with the model, while the integration of cosine similarity metrics ensures that the results are both relevant and precise.

PIPELINE / FLOWCHART:

\- Check GPU availability using \`torch.cuda\`  
  \- If GPU is available, set device to GPU  
  \- If GPU is not available, set device to CPU

\- Enable mixed precision and optimize GPU usage  
  \- Set default dtype to \`float16\`  
  \- Enable TF32 for faster matrix multiplication  
  \- Enable cuDNN benchmark for performance optimization

\- Install required libraries and tools  
  \- \`llama\_parse\`, \`huggingface\_hub\`, \`langchain\`, \`gradio\`, etc.  
  \- Install additional dependencies like \`nltk\` for text processing

\- Preprocessing and parsing pipeline  
  \- Parse documents using \`LlamaParse\` or similar tools  
  \- Extract key elements (e.g., chapter titles, definitions, tables)  
  \- Save parsed results for further processing

\- Document chunking and embedding generation  
  \- Split parsed documents into chunks using \`RecursiveCharacterTextSplitter\`  
  \- Generate embeddings for the chunks using the BGE model on GPU  
  \- Ensure embeddings are contiguous and in correct format

\- Initialize vector store with FAISS  
  \- Use FAISS to create an index for fast similarity search  
  \- Add document embeddings to the FAISS index

\- Document retrieval and similarity search  
  \- Retrieve top-k most similar documents for a given query  
  \- Compute similarity scores using cosine similarity

\- Text generation with Qwen LLM pipeline  
  \- Format query with prompt template  
  \- Use Qwen LLM to generate responses based on retrieved context

\- Gradio interface for querying the RAG system  
  \- Enable real-time user input and response display  
  \- Provide a simple interface for querying the system

\- Metrics and visualization for system evaluation  
  \- Measure performance with similarity scores and retrieval times  
  \- Visualize normalized similarity scores using bar charts

