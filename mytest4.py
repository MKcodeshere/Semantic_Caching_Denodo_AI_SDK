import requests
import json
import os
import time
import faiss
import numpy as np
import base64
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Configuration
DENODO_AI_SDK_URL = "http://localhost:8008"  # Update with your Denodo AI SDK URL
DATA_CATALOG_BASE_URL = "http://localhost:39090/denodo-data-catalog"  # Update with your Data Catalog URL
DATA_CATALOG_SERVER_ID = 1  # Default is 1, update if different
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
SIMILARITY_THRESHOLD = 0.90  # Similarity threshold for matching questions
AUTH_USERNAME = "admin"  # Replace with your Denodo username
AUTH_PASSWORD = "admin"  # Replace with your Denodo password
DATA_CATALOG_VERIFY_SSL = True  # Set to False if using self-signed certificates


class SemanticCache:
    def __init__(self, embedding_model, similarity_threshold=0.90):
        """Initialize the semantic cache with an embedding model and similarity threshold."""
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache = {
            "questions": [],
            "embeddings": None,
            "sql_queries": [],
            "results": []
        }
        self.faiss_index = None
        self.initialize_faiss()
    
    def initialize_faiss(self):
        """Initialize an empty FAISS index."""
        # text-embedding-3-large has 3072 dimensions
        embedding_dim = 3072
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
    def add_to_cache(self, question, sql_query, result):
        """Add a question, SQL query, and result to the cache."""
        # Get embedding for the question
        question_embedding = self.embedding_model.embed_query(question)
        
        # Add to cache
        self.cache["questions"].append(question)
        self.cache["sql_queries"].append(sql_query)
        self.cache["results"].append(result)
        
        # Update FAISS index
        if self.faiss_index.ntotal == 0:
            # First time adding, initialize with the first vector
            self.cache["embeddings"] = np.array([question_embedding], dtype=np.float32)
            self.faiss_index.add(self.cache["embeddings"])
        else:
            # Add to existing index
            new_embedding = np.array([question_embedding], dtype=np.float32)
            self.cache["embeddings"] = np.vstack([self.cache["embeddings"], new_embedding])
            # Recreate the index (simple approach for prototype)
            self.faiss_index = faiss.IndexFlatL2(self.cache["embeddings"].shape[1])
            self.faiss_index.add(self.cache["embeddings"])
        
        print(f"Added to cache: {question}")
        
    def find_similar_question(self, question):
        """
        Find a similar question in the cache.
        Returns the question, SQL query, result, and similarity if found, otherwise None values.
        """
        if self.faiss_index.ntotal == 0:
            return None, None, None, 0.0
        
        # Get embedding for the question
        question_embedding = self.embedding_model.embed_query(question)
        query_vector = np.array([question_embedding], dtype=np.float32)
        
        # Search for similar questions
        distances, indices = self.faiss_index.search(query_vector, 1)
        
        # Calculate similarity from L2 distance
        max_distance = 20  # Arbitrary large number for normalization
        similarity = 1 - (distances[0][0] / max_distance)
        
        if similarity >= self.similarity_threshold and indices[0][0] < len(self.cache["questions"]):
            index = indices[0][0]
            return self.cache["questions"][index], self.cache["sql_queries"][index], self.cache["results"][index], similarity
        
        return None, None, None, 0.0


class DenodoAISDKClient:
    def __init__(self, base_url, auth_username, auth_password):
        """Initialize the Denodo AI SDK client."""
        self.base_url = base_url
        self.auth = (auth_username, auth_password)
    
    def answer_data_question(self, question, use_views=None):
        """
        Send a question to the Denodo AI SDK answerDataQuestion endpoint.
        This endpoint specifically targets data questions (SQL generation).
        """
        endpoint = f"{self.base_url}/answerDataQuestion"
        
        payload = {
            "question": question,
            "verbose": True,
            "disclaimer": False
        }
        
        if use_views:
            payload["use_views"] = use_views
        
        start_time = time.time()
        response = requests.post(endpoint, json=payload, auth=self.auth)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = end_time - start_time
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None


def execute_vql(vql, auth, data_catalog_url=DATA_CATALOG_BASE_URL, server_id=DATA_CATALOG_SERVER_ID, 
              limit=1000, verify_ssl=DATA_CATALOG_VERIFY_SSL):
    """
    Execute VQL against Data Catalog with support for OAuth token or Basic auth.
    
    Args:
        vql: VQL query to execute
        auth: Either (username, password) tuple for basic auth or OAuth token string
        data_catalog_url: Base URL for Data Catalog
        server_id: Server identifier
        limit: Maximum number of rows to return
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        Tuple of (status_code, result or error_message)
    """
    # Clean up the query before execution
    cleaned_vql = vql.strip()
    
    # Remove markdown SQL code blocks if present
    if cleaned_vql.startswith("```sql"):
        cleaned_vql = cleaned_vql.replace("```sql", "", 1)
    if cleaned_vql.startswith("```"):
        cleaned_vql = cleaned_vql.replace("```", "", 1)
    if cleaned_vql.endswith("```"):
        cleaned_vql = cleaned_vql[:-3]
    
    cleaned_vql = cleaned_vql.strip()
    
    print(f"Executing VQL: {cleaned_vql}")
    
    # Prepare execution URL
    execution_url = f"{data_catalog_url}/public/api/askaquestion/execute"
    
    # Prepare headers based on auth type
    headers = {'Content-Type': 'application/json'}
    if isinstance(auth, tuple):
        # Basic auth
        auth_str = f"{auth[0]}:{auth[1]}"
        encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
        headers['Authorization'] = f'Basic {encoded_auth}'
    else:
        # OAuth token
        headers['Authorization'] = f'Bearer {auth}'
    
    data = {
        "vql": cleaned_vql,
        "limit": limit
    }
    
    try:
        response = requests.post(
            f"{execution_url}?serverId={server_id}",
            json=data,
            headers=headers,
            verify=verify_ssl
        )
        response.raise_for_status()
        
        json_response = response.json()
        if not json_response.get('rows'):
            print("Query returned no rows")
            return response.status_code, []
        
        print("Query executed successfully")
        # Extract rows from the response
        rows = json_response.get('rows', [])
        return response.status_code, rows
    
    except requests.HTTPError as e:
        error_message = f"HTTP Error: {e}"
        try:
            error_response = e.response.json()
            error_message = f"Data Catalog error: {error_response.get('message', 'No details')}"
        except:
            pass
        print(f"Error executing VQL: {error_message}")
        return e.response.status_code, error_message
    
    except requests.RequestException as e:
        error_message = f"Failed to connect to the server: {str(e)}"
        print(f"{error_message}. VQL: {cleaned_vql}")
        return 500, error_message


def are_questions_semantically_related(question1, question2):
    """
    Use an LLM to determine if two questions are semantically related enough
    to potentially reuse and modify the SQL from one to answer the other.
    
    Returns: 
        tuple: (boolean indicating if related, explanation string)
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_template("""
    I need to determine if two questions about a banking database are semantically related enough 
    that the SQL query for one could be modified to answer the other. The database contains 
    information about bank accounts, customers, loans, properties, and loan officers.
    
    Question 1: {question1}
    Question 2: {question2}
    
    First, analyze what each question is asking for:
    - What entity/table is each question about? (customers, loans, accounts, properties, etc.)
    - What operation is being performed? (counting, summing, averaging, ranking, listing, etc.)
    - What filters, groupings, or sorting conditions are applied?
    - What are the core parameters that might change (like top N, time period, location, etc.)?
    
    Then determine:
    1. Are the questions asking about the same entity/table or joined tables?
    2. Are they performing similar operations that could be modified?
    3. Are the differences between the questions mainly in parameters, filters, or minor variations?
    4. Would the SQL structure be largely the same with only minor changes to clauses like WHERE, LIMIT, ORDER BY?
    
    Output your decision as a JSON object with these fields:
    {{
      "are_related": true/false,
      "explanation": "Brief explanation of your reasoning",
      "primary_entity": "The main entity/table being queried",
      "operation_type": "The type of operation (count, sum, average, rank, list, etc.)",
      "parameter_differences": "Description of any parameter differences that would need modification"
    }}
    
    The questions are related if they query the same primary entity/entities and perform operations that are similar enough
    that one SQL query could be modified to answer the other question with relatively minor changes.
    """)
    
    response = llm.invoke(prompt.format(question1=question1, question2=question2))
    
    try:
        # Parse the response to extract the JSON
        content = response.content
        
        # Find the JSON object in the response (it might be in a code block or mixed with text)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
            return analysis.get("are_related", False), analysis.get("explanation", "")
        else:
            return False, "Failed to parse LLM response"
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response.content}")
        return False, f"Error in analysis: {str(e)}"


def modify_sql_query(original_query, new_question, original_question):
    """
    Use an LLM to modify the original SQL query based on the differences between
    the original question and the new question.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_template("""
    I have a SQL query that was generated for a specific question. I need to modify this SQL query
    for a new, semantically similar question.
    
    Original question: {original_question}
    Original SQL query: {original_query}
    
    New question: {new_question}
    
    Please analyze the differences between the questions and modify the SQL query accordingly. Common changes include:
    - Different numerical parameters (e.g., top 5 vs top 10)
    - Different time periods or date ranges
    - Different sorting requirements
    - Different filtering conditions 
    - Different but equivalent column references
    
    Look at the original SQL to understand the schema and table structure. Only modify what's necessary. For example for state values it should be always code, even when the question contains names like California, Newyork, Texas the filter should be modified only with state code for example state code of california is CA.
    
    Only output the modified SQL query, with no additional text, formatting, markdown, or code blocks.
    If no changes are needed, return the original query exactly.
    
    DO NOT include any ```sql or ``` markers or any other markdown formatting in your response. 
    Return ONLY the raw SQL query.
    """)
    
    chain = prompt | llm | StrOutputParser()
    modified_query = chain.invoke({
        "original_question": original_question,
        "original_query": original_query,
        "new_question": new_question
    })
    
    # Clean up any markdown formatting that might have been included
    modified_query = modified_query.strip()
    
    # Remove markdown SQL code blocks if present
    if modified_query.startswith("```sql"):
        modified_query = modified_query.replace("```sql", "", 1)
    if modified_query.startswith("```"):
        modified_query = modified_query.replace("```", "", 1)
    if modified_query.endswith("```"):
        modified_query = modified_query[:-3]
        
    return modified_query.strip()


def main():
    # Initialize OpenAI embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072,
        api_key=OPENAI_API_KEY
    )
    
    # Initialize the semantic cache
    cache = SemanticCache(embeddings, SIMILARITY_THRESHOLD)
    
    # Initialize the Denodo AI SDK client
    denodo_client = DenodoAISDKClient(DENODO_AI_SDK_URL, AUTH_USERNAME, AUTH_PASSWORD)
    
    # Auth tuple for Data Catalog executions
    auth = (AUTH_USERNAME, AUTH_PASSWORD)
    
    # Track performance metrics
    performance_results = {
        "questions_processed": 0,
        "cache_hits": 0,
        "sdk_calls": 0,
        "cache_response_times": [],
        "sdk_response_times": [],
        "question_sources": []  # 'cache' or 'sdk' for each question
    }
    
    # Define a set of questions for testing that showcase the semantic cache capabilities
    test_questions = [
        # Set 1: Loan amount queries - demonstrating semantic similarity
        "What is the total loan amount across all loans?",
        "Sum up the value of all loans in the bank",
        "Calculate the aggregate loan amount in our portfolio",
        
        # Set 2: Customer location queries - similar intent, different wording
        "How many customers do we have in the state of CA?",
        "Count the number of clients in NewYork",
        "What's the total count of customers residing in CA?",
        
        # Set 3: Account balance queries - variations in wording
        "What is the average balance of checking accounts?",
        "Calculate the mean balance for accounts of type 'Checking'",
        "What's the typical balance in checking accounts?",
        
        # Set 4: Top N patterns - same query structure with different N parameter
        "Who are the top 5 customers with the highest loan amounts?",
        "Who are the top 10 customers with the highest loan amounts?",
        "Who are the top 3 customers with the highest loan amounts?",
        
        # Set 5: Complex JOIN queries involving properties and loans
        "Provide me the list of approved mortgages including property values",
        "Show all approved loans with their associated property information",
        "List all approved mortgages along with the value of the properties",
        
        # Set 6: Complex sorting requirements - multi-level ordering
        "Please provide a list of customers ordered by their loan amounts in descending order. If there are same amounts order by their last name",
        "Sort customers by total loan amount (highest first) and then alphabetically by last name for ties",
        "Rank borrowers by loan size, with alphabetical ordering by surname when amounts match"
    ]
    
    # Process each question
    for question in test_questions:
        print(f"\n\n{'='*80}")
        print(f"PROCESSING QUESTION: '{question}'")
        print(f"{'='*80}")
        
        # Check if a similar question exists in the cache
        cached_question, cached_sql, cached_result, similarity = cache.find_similar_question(question)
        
        if cached_question:
            print(f"✓ FOUND SIMILAR QUESTION IN CACHE: '{cached_question}'")
            print(f"✓ SIMILARITY SCORE: {similarity:.2f}")
            
            # Use LLM to validate if the questions are truly semantically related
            are_related, explanation = are_questions_semantically_related(cached_question, question)
            
            if are_related:
                print(f"✓ LLM VALIDATION: Questions are semantically related")
                print(f"✓ EXPLANATION: {explanation}")
                
                # Modify the SQL query for the new question
                start_time = time.time()
                modified_sql = modify_sql_query(cached_sql, question, cached_question)
                
                print(f"\n📜 CACHE MODIFICATION:")
                print(f"Original SQL: {cached_sql}")
                print(f"Modified SQL: {modified_sql}")
                
                # Execute the modified SQL directly using Data Catalog
                status_code, result = execute_vql(modified_sql, auth)
                processing_time = time.time() - start_time
                
                # Update performance metrics
                performance_results["cache_response_times"].append(processing_time)
                performance_results["question_sources"].append("cache")
                performance_results["cache_hits"] += 1
                performance_results["questions_processed"] += 1
                
                if 200 <= status_code < 300:
                    print(f"\n✅ CACHE HIT: SQL execution successful")
                    print(f"⏱️ RESPONSE TIME: {processing_time:.2f} seconds")
                    print(f"💰 ESTIMATED TOKEN SAVINGS: ~1000 tokens")
                    print(f"🔍 RESULT: {result[:3] if len(result) > 3 else result}...")
                    
                    query_explanation = f"Query that {question}"  # Simple explanation based on the question
                    cache.add_to_cache(question, modified_sql, query_explanation)
                else:
                    print(f"\n❌ CACHE MISS: SQL execution failed: {result}")
                    print(f"⚠️ Falling back to Denodo AI SDK...")
                    
                    # Fall back to AI SDK
                    denodo_start = time.time()
                    sdk_result = denodo_client.answer_data_question(question)
                    sdk_time = time.time() - denodo_start
                    
                    # Update performance metrics
                    performance_results["sdk_response_times"].append(sdk_time)
                    performance_results["sdk_calls"] += 1
                    performance_results["question_sources"][-1] = "sdk"  # Update the source for this question
                    
                    if sdk_result:
                        sql_query = sdk_result.get('sql_query', '')
                        query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")

                        print(f"📊 SQL FROM SDK: {sql_query}")
                        print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                        
                        # Add to cache
                        cache.add_to_cache(question, sql_query, query_explanation)
                    else:
                        print("❌ FAILED: No response from Denodo AI SDK")
            else:
                print(f"❌ LLM VALIDATION: Questions are not semantically related")
                print(f"ℹ️ EXPLANATION: {explanation}")
                print(f"⚠️ Falling back to Denodo AI SDK...")
                
                # Fall back to AI SDK
                denodo_start = time.time()
                sdk_result = denodo_client.answer_data_question(question)
                sdk_time = time.time() - denodo_start
                
                # Update performance metrics
                performance_results["sdk_response_times"].append(sdk_time)
                performance_results["sdk_calls"] += 1
                performance_results["questions_processed"] += 1
                performance_results["question_sources"].append("sdk")
                
                if sdk_result:
                    sql_query = sdk_result.get('sql_query', '')
                    query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")
                    
                    print(f"📊 SQL FROM SDK: {sql_query}")
                    print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                    
                    # Add to cache
                    cache.add_to_cache(question, sql_query, query_explanation)
                else:
                    print("❌ FAILED: No response from Denodo AI SDK")
        else:
            print("ℹ️ No similar question found in cache")
            print("🔄 Querying Denodo AI SDK...")
            
            # Send the question to the Denodo AI SDK
            denodo_start = time.time()
            sdk_result = denodo_client.answer_data_question(question)
            sdk_time = time.time() - denodo_start
            
            # Update performance metrics
            performance_results["sdk_response_times"].append(sdk_time)
            performance_results["sdk_calls"] += 1
            performance_results["questions_processed"] += 1
            performance_results["question_sources"].append("sdk")
            
            if sdk_result:
                sql_query = sdk_result.get('sql_query', '')
                query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")
                
                print(f"📊 SQL FROM SDK: {sql_query}")
                print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                
                # Add to cache
                cache.add_to_cache(question, sql_query, query_explanation)
            else:
                print("❌ FAILED: No response from Denodo AI SDK")
    
    # Generate visual performance report
    print("\n\n" + "="*80)
    print("                  SEMANTIC CACHE PERFORMANCE REPORT                  ")
    print("="*80)
    
    # Calculate aggregate statistics
    avg_cache_time = sum(performance_results["cache_response_times"]) / len(performance_results["cache_response_times"]) if performance_results["cache_response_times"] else 0
    avg_sdk_time = sum(performance_results["sdk_response_times"]) / len(performance_results["sdk_response_times"]) if performance_results["sdk_response_times"] else 0
    time_saved = avg_sdk_time * performance_results["cache_hits"] - avg_cache_time * performance_results["cache_hits"]
    speedup = avg_sdk_time/avg_cache_time if avg_cache_time > 0 else 0
    
    # Create a summary section
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"{'Total Questions Processed:':<40} {performance_results['questions_processed']}")
    print(f"{'Cache Hits:':<40} {performance_results['cache_hits']} ({(performance_results['cache_hits']/performance_results['questions_processed'])*100:.1f}%)")
    print(f"{'SDK Calls Required:':<40} {performance_results['sdk_calls']}")
    print(f"{'SDK Calls Avoided:':<40} {performance_results['cache_hits']}")
    
    # Performance metrics
    print(f"\n⏱️ RESPONSE TIME COMPARISON:")
    print(f"{'Average SDK Response Time:':<40} {avg_sdk_time:.2f} seconds")
    print(f"{'Average Cache Response Time:':<40} {avg_cache_time:.2f} seconds")
    print(f"{'Speed Improvement:':<40} {speedup:.1f}x faster with cache")
    print(f"{'Total Time Saved:':<40} {time_saved:.2f} seconds")
    
    # Cost savings
    token_savings = performance_results["cache_hits"] * 1000  # Assuming 1000 tokens saved per cache hit
    print(f"\n💰 COST ANALYSIS:")
    print(f"{'Estimated Token Savings:':<40} {token_savings:,} tokens")
    print(f"{'Estimated Cost Savings:':<40} ${token_savings/1000 * 0.001:.2f} (@ $0.001 per 1K tokens)")
    
    # Create a visual performance chart
    print("\n📈 PERFORMANCE VISUALIZATION:")
    max_time = max(
        max(performance_results["cache_response_times"] or [0]), 
        max(performance_results["sdk_response_times"] or [0])
    )
    scale_factor = 50 / max_time if max_time > 0 else 1
    
    for i, question in enumerate(test_questions[:performance_results["questions_processed"]]):
        q_summary = question[:60] + "..." if len(question) > 60 else question
        source = performance_results["question_sources"][i] if i < len(performance_results["question_sources"]) else "unknown"
        
        print(f"\nQ{i+1}: {q_summary}")
        
        if source == "cache":
            cache_time = performance_results["cache_response_times"][performance_results["question_sources"][:i+1].count("cache")-1]
            cache_bar = "█" * int(cache_time * scale_factor)
            print(f"CACHE: {cache_bar} {cache_time:.2f}s")
        elif source == "sdk":
            sdk_time = performance_results["sdk_response_times"][performance_results["question_sources"][:i+1].count("sdk")-1]
            sdk_bar = "█" * int(sdk_time * scale_factor)
            print(f"SDK:   {sdk_bar} {sdk_time:.2f}s")
    
    # Display cache contents
    print("\n\n" + "="*80)
    print("                  SEMANTIC CACHE CONTENTS                  ")
    print("="*80)
    print(f"Number of cached questions: {len(cache.cache['questions'])}")
    for i, question in enumerate(cache.cache["questions"]):
        sql = cache.cache["sql_queries"][i]
        sql_summary = sql[:80] + "..." if len(sql) > 80 else sql
        print(f"{i+1}. '{question}'")
        print(f"   SQL: {sql_summary}")
        print("-" * 60)
        
if __name__ == "__main__":
    main()