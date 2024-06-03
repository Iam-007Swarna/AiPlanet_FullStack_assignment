# Astronomy Black Hole Hybrid RAG

This repository contains a Python codebase for a hybrid retrieval-augmented generation (RAG) system designed for the astronomy domain, with a focus on black holes. The system combines various retrieval techniques, including dense vector retrieval, BM25 retrieval, and knowledge graph retrieval, to provide comprehensive and accurate responses to user queries.

## Features

- **Hybrid Retrieval**: The system utilizes a hybrid retrieval approach that combines dense vector retrieval, BM25 retrieval, and knowledge graph retrieval to leverage the strengths of each technique.
- **Query Fusion**: The system employs query fusion to generate multiple search queries from a single input query, improving the coverage and relevance of retrieved information.
- **Reranking**: A sentence transformer-based reranker is used to improve the ranking of retrieved documents and passages.
- **Chat Engine**: The system includes a chat engine that provides conversational responses to user queries, leveraging the retrieved information and the language model's capabilities.
- **Reference Document Tracking**: The chat engine's responses include references to the relevant documents and page numbers used to generate the response.

## Installation

1. Clone the repository:


git clone https://github.com/your-repo/astronomy-bh-hybrid-rag.git


2. Install the required dependencies:


pip install -r requirements.txt


3. Set up the necessary environment variables:


export HF_TOKEN=your_huggingface_token
export MISTRAL_API=your_mistral_api_key


## Usage

1. Run the Jupyter Notebook or Python script containing the codebase.
2. Use the `get_query` function to submit your query to the system:


result_hybrid, result_kg, result_hybrid_bm25 = get_query("your query here")


The function will return three responses:
- `result_hybrid`: The response from the hybrid retrieval system.
- `result_kg`: The response from the knowledge graph retrieval system.
- `result_hybrid_bm25`: The response from the hybrid BM25 retrieval system.

Each response includes the generated answer and references to the relevant documents and page numbers.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for bug fixes, improvements, or new features.

## License

This project is licensed under GNU Affero General Public License v3.0 (AGPL-3.0).
