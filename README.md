# Semantic_Caching_Denodo_AI_SDK

# Semantic Cache for Denodo AI SDK

This repository provides an implementation of a semantic cache to improve efficiency, reduce costs, and enhance performance when using the Denodo AI SDK for natural language queries on structured data.

## Overview

The semantic cache system stores natural language questions, their corresponding SQL queries, and query explanations. When a new question is asked, the system checks if a semantically similar question exists in the cache. If found, it modifies the existing SQL query rather than generating a new one from scratch, resulting in faster response times and lower costs.

This implementation follows the process described in research showing that modifying existing SQL is more efficient than generating new SQL, similar to how software engineers reuse and adapt existing code.

## Key Features

- **Vector-based Semantic Matching**: Uses OpenAI embeddings to find similar questions
- **LLM Validation**: Uses LLM reasoning to confirm questions are truly semantically related
- **SQL Modification**: Intelligently modifies cached SQL for new but similar questions
- **Fallback Mechanism**: If modification fails, falls back to the standard SDK
- **Schema Agnostic**: Works with any database schema without domain-specific knowledge

## Components

1. **SemanticCache Class**: Core component that stores questions, SQL queries, and explanations
2. **DenodoAISDKClient**: Wrapper for interacting with Denodo AI SDK
3. **are_questions_semantically_related**: Uses LLM to validate if questions are semantically related
4. **modify_sql_query**: Uses LLM to modify SQL queries for new questions
5. **execute_vql**: Executes modified queries against Data Catalog

## Example Use Cases

The semantic cache is particularly effective for:

1. **Parameter Variations**: Questions like "Who are the top 5 customers with the highest loan amounts?" vs "Who are the top 10 customers with the highest loan amounts?"

2. **Wording Variations**: Questions like "How many customers do we have in the state of CA?" vs "Count the number of clients in California"

3. **Complex Queries**: Including those requiring joins or complex sorting, where reuse significantly reduces cost

## Benefits

- **Performance**: Response times can be up to 5x faster for cached queries
- **Cost Reduction**: Uses smaller/fewer LLM calls when modifying existing SQL
- **Consistency**: Similar questions get consistently structured SQL
- **Accuracy**: Schema-aware modifications with validation ensure correctness

## Configuration

Key configuration parameters:
- `DENODO_AI_SDK_URL`: URL of your Denodo AI SDK
- `DATA_CATALOG_BASE_URL`: URL of your Data Catalog
- `OPENAI_API_KEY`: Your OpenAI API key
- `SIMILARITY_THRESHOLD`: Threshold for determining similarity (default: 0.90)

## Usage

```python
# Initialize the semantic cache
cache = SemanticCache(embeddings, SIMILARITY_THRESHOLD)

# Initialize the Denodo AI SDK client
denodo_client = DenodoAISDKClient(DENODO_AI_SDK_URL, AUTH_USERNAME, AUTH_PASSWORD)

# Process natural language question
question = "Who are the top 5 customers with the highest loan amounts?"

# Check if similar question exists in cache
cached_question, cached_sql, cached_explanation, similarity = cache.find_similar_question(question)

if cached_question and are_semantically_related:
    # Modify SQL for new question
    modified_sql = modify_sql_query(cached_sql, question, cached_question)
    # Execute modified SQL
    results = execute_vql(modified_sql, auth)
else:
    # Fall back to SDK
    sdk_result = denodo_client.answer_data_question(question)
    # Cache the result
    cache.add_to_cache(question, sdk_result.get('sql_query', ''), sdk_result.get('query_explanation', ''))
```

## Installation

1. Clone this repository
2. Install requirements:
```
pip install -r requirements.txt
```
3. Configure the environment variables
4. Run the example:
```
python semantic_cache.py
```

## Requirements

- Python 3.10+
- LLM access ( Open source or ollama small models)
- Denodo Platform 9.0.5 or higher
- Denodo AI SDK Installed ( run only api ) 
- FAISS for vector storage
- LangChain for embeddings and LLM interfaces

## Future Improvements

- Integration with other vector databases (Pinecone, Weaviate)
- Automated question caching based on user feedback
- More sophisticated SQL modification strategies

This implementation demonstrates how semantic caching can provide significant improvements in performance and cost-efficiency when using the Denodo AI SDK for natural language queries on structured data.
