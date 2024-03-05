```markdown
# S3LLM - Simplifying Scientific Software with LLMs

## Overview

S3LLM is an integral component of the HOLOSCAN project, designed to revolutionize the way we interact with large-scale scientific software. It employs advanced Large Language Models (LLMs), specifically leveraging the open-source LLaMA-2 models, to facilitate an intuitive, conversational interface for exploring and understanding complex codebases. This innovative tool transforms natural language queries into Feature Query Language (FQL) queries, enabling efficient navigation and comprehension of extensive software ecosystems, including their code, metadata, and associated documentation.

## Abstract

The S3LLM framework addresses the intricate challenges posed by large-scale scientific software, characterized by its vast codebase and sophisticated computing architectures. By harnessing generative AI and LLMs, S3LLM offers a groundbreaking approach to decode and analyze scientific codes interactively and efficiently. It combines code analysis, metadata inspection, and technical documentation review in a unified, user-friendly platform, making the understanding of complex scientific software accessible to a broader range of users.

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
   ```
