## *S3LLM*: Large-*S*cale *S*cientific *S*oftware Understanding with *LLMs* using Source, Metadata, and Document

## Abstract

The understanding of large-scale scientific software poses significant challenges due to its diverse codebase, extensive code length, and target computing architectures. The emergence of generative AI, specifically large language models (LLMs), provides novel pathways for understanding such complex scientific codes. The proposed S3LLM, an LLM-based framework designed to enable the examination of source code, code metadata, and summarized information in conjunction with textual technical reports in an interactive, conversational manner through a user-friendly interface. In particular, S3LLM utilizes open-source LLaMA-2 models to improve code analysis by converting natural language queries into Feature Query Language (FQL) queries, facilitating the quick scanning and parsing of entire code repositories. In addition, S3LLM is equipped to handle diverse metadata types, including DOT, SQL, and customized formats. Furthermore, S3LLM incorporates retrieval augmented generation (RAG) and LangChain technologies to directly query extensive documents. S3LLM demonstrates the potential of using locally deployed open-source LLMs for the rapid understanding of large-scale scientific computing software, eliminating the need for extensive coding expertise, and thereby making the process more efficient and effective.

[Paper: https://arxiv.org/abs/2403.10588](https://arxiv.org/abs/2403.10588)

## Installation

Ensure you have Python 3.9 or higher installed on your machine. Follow these steps to set up S3LLM:

1. Clone the repository or download the source code.
2. Navigate to the project directory and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you encounter any issues with GPU connectivity, try reinstalling the transformers library with the following command:

   ```bash
   !CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers
   !CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
   ```
