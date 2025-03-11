import requests
import json
import os
import time
import faiss
import numpy as np
import base64
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch
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
VISUALIZATION_OUTPUT_DIR = "cache_analysis_visualizations"  # Directory to save visualizations


# Create output directory for visualizations if it doesn't exist
if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
    os.makedirs(VISUALIZATION_OUTPUT_DIR)


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
    I need to determine if two questions about a database are semantically related enough 
    that the SQL query for one could be modified to answer the other.
    
    Question 1: {question1}
    Question 2: {question2}
    
    First, analyze what each question is asking for:
    - What entity/table is each question about? (e.g., customers, products, transactions, etc.)
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


# Matplotlib visualization functions
def visualize_token_usage(performance_results):
    """Generate a bar chart for token usage comparison."""
    # Calculate total cache tokens
    total_cache_tokens = performance_results["token_usage"]["cache_validation_tokens"] + performance_results["token_usage"]["cache_modification_tokens"]
    
    # Calculate estimated token savings
    avg_sdk_tokens = performance_results["token_usage"]["sdk_tokens"] / performance_results["sdk_calls"] if performance_results["sdk_calls"] > 0 else 0
    tokens_saved = (avg_sdk_tokens * performance_results["cache_hits"]) - total_cache_tokens
    
    # Prepare data for visualization
    categories = ['SDK Total', 'Cache Validation', 'Cache Modification', 'Total Cache Usage', 'Estimated Savings']
    values = [
        performance_results["token_usage"]["sdk_tokens"],
        performance_results["token_usage"]["cache_validation_tokens"],
        performance_results["token_usage"]["cache_modification_tokens"],
        total_cache_tokens,
        tokens_saved if tokens_saved > 0 else 0
    ]
    
    # Set up the figure with a specific size
    plt.figure(figsize=(12, 6))
    
    # Create a horizontal bar chart with a colorful palette
    bars = plt.barh(categories, values, color=sns.color_palette("viridis", len(categories)))
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        plt.text(label_x_pos + 100, bar.get_y() + bar.get_height()/2, f'{int(width):,}', 
                 va='center', fontweight='bold')
    
    # Add labels and title with improved styling
    plt.xlabel('Number of Tokens', fontsize=12, fontweight='bold')
    plt.title('Token Usage Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/token_usage.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/token_usage.png"


def visualize_response_times(performance_results):
    """Generate a bar chart for response time comparison."""
    # Calculate average times
    avg_cache_time = sum(performance_results["cache_response_times"]) / len(performance_results["cache_response_times"]) if performance_results["cache_response_times"] else 0
    avg_sdk_time = sum(performance_results["sdk_response_times"]) / len(performance_results["sdk_response_times"]) if performance_results["sdk_response_times"] else 0
    time_saved = avg_sdk_time * performance_results["cache_hits"] - avg_cache_time * performance_results["cache_hits"]
    
    # Prepare data for visualization
    categories = ['Average SDK Response', 'Average Cache Response', 'Total Time Saved']
    values = [avg_sdk_time, avg_cache_time, time_saved if time_saved > 0 else 0]
    
    # Set up the figure with a specific size
    plt.figure(figsize=(12, 5))
    
    # Create a horizontal bar chart with a colorful palette
    bars = plt.barh(categories, values, color=sns.color_palette("cool", len(categories)))
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        plt.text(label_x_pos + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', 
                 va='center', fontweight='bold')
    
    # Add labels and title with improved styling
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Response Time Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Calculate speedup for annotation
    speedup = avg_sdk_time/avg_cache_time if avg_cache_time > 0 else 0
    plt.figtext(0.5, 0.01, f'Cache is {speedup:.1f}x faster than SDK calls', 
                ha='center', fontsize=12, fontweight='bold')
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/response_times.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/response_times.png"


def visualize_question_processing_times(performance_results):
    """Generate a bar chart for individual question processing times."""
    # Get the data
    question_numbers = [f"Q{i+1}" for i in range(len(performance_results["question_texts"]))]
    times = performance_results["all_response_times"]
    sources = performance_results["question_sources"]
    
    # Set up the figure with a specific size based on number of questions
    plt.figure(figsize=(12, max(6, len(question_numbers) * 0.5)))
    
    # Create color mapping for sources
    colors = {'cache': 'forestgreen', 'sdk': 'orangered'}
    bar_colors = [colors[source] for source in sources]
    
    # Create horizontal bar chart
    bars = plt.barh(question_numbers, times, color=bar_colors)
    
    # Add value labels and source indicators
    for i, (bar, source) in enumerate(zip(bars, sources)):
        width = bar.get_width()
        source_label = "CACHE" if source == 'cache' else "SDK"
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}s ({source_label})', 
                 va='center')
    
    # Add labels and title
    plt.xlabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Question Number', fontsize=12, fontweight='bold')
    plt.title('Question Processing Times', fontsize=16, fontweight='bold', pad=20)
    
    # Create a custom legend
    legend_elements = [
        Patch(facecolor=colors['cache'], label='Cache'),
        Patch(facecolor=colors['sdk'], label='SDK')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Calculate and add average times for each source
    cache_times = [t for i, t in enumerate(times) if sources[i] == 'cache']
    sdk_times = [t for i, t in enumerate(times) if sources[i] == 'sdk']
    
    avg_cache = sum(cache_times) / len(cache_times) if cache_times else 0
    avg_sdk = sum(sdk_times) / len(sdk_times) if sdk_times else 0
    
    annotation_text = (f"Average Times:\n"
                      f"Cache: {avg_cache:.2f}s\n"
                      f"SDK: {avg_sdk:.2f}s")
    
    # Position the annotation in the lower right corner
    plt.annotate(annotation_text, xy=(0.95, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 ha='right', va='bottom')
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/question_times.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/question_times.png"


def visualize_cache_effectiveness(performance_results):
    """Generate a pie chart for cache hit rate."""
    # Calculate rates
    cache_hit_rate = (performance_results["cache_hits"] / performance_results["questions_processed"]) * 100 if performance_results["questions_processed"] > 0 else 0
    sdk_call_rate = 100 - cache_hit_rate
    
    # Data for visualization
    labels = ['Cache Hits', 'SDK Calls']
    sizes = [cache_hit_rate, sdk_call_rate]
    
    # Use a vibrant color scheme
    colors = ['#4CAF50', '#FF5722']
    
    # Create a figure with a specific size
    plt.figure(figsize=(8, 8))
    
    # Create a pie chart with a slight explosion for the first slice
    explode = (0.1, 0)  # explode the 1st slice (Cache Hits)
    
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, 
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    
    # Customize the pie chart
    plt.setp(autotexts, size=12, weight="bold")
    
    # Add a title
    plt.title('Cache Effectiveness', fontsize=16, fontweight='bold', pad=20)
    
    # Add some statistics as annotations
    stats_text = (f"Total Questions: {performance_results['questions_processed']}\n"
                 f"Cache Hits: {performance_results['cache_hits']}\n"
                 f"SDK Calls Required: {performance_results['sdk_calls']}\n"
                 f"SDK Calls Avoided: {performance_results['cache_hits']}")
    
    plt.annotate(stats_text, xy=(-0.2, -0.15), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/cache_effectiveness.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/cache_effectiveness.png"


def visualize_similarity_distribution(cache):
    """Generate a histogram of similarity scores for all cached questions."""
    if not cache.cache["questions"] or len(cache.cache["questions"]) < 2:
        return None  # Not enough data
    
    # Calculate similarities between all pairs of questions
    questions = cache.cache["questions"]
    similarities = []
    
    for i in range(len(questions)):
        for j in range(i+1, len(questions)):
            # Get embeddings
            embedding_i = cache.cache["embeddings"][i]
            embedding_j = cache.cache["embeddings"][j]
            
            # Calculate cosine similarity (normalized dot product)
            dot_product = np.dot(embedding_i, embedding_j)
            norm_i = np.linalg.norm(embedding_i)
            norm_j = np.linalg.norm(embedding_j)
            similarity = dot_product / (norm_i * norm_j)
            
            similarities.append(similarity)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, bins=20, kde=True)
    
    # Add a vertical line at the threshold value
    plt.axvline(x=cache.similarity_threshold, color='r', linestyle='--', 
                label=f'Threshold ({cache.similarity_threshold})')
    
    # Add labels and title
    plt.xlabel('Similarity Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Distribution of Similarity Scores Between Cached Questions', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend()
    
    # Add grid for better readability
    plt.grid(linestyle='--', alpha=0.7)
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/similarity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/similarity_distribution.png"


def visualize_cost_analysis(performance_results):
    """Generate a visualization of cost savings."""
    # Calculate token metrics
    total_cache_tokens = performance_results["token_usage"]["cache_validation_tokens"] + performance_results["token_usage"]["cache_modification_tokens"]
    avg_sdk_tokens = performance_results["token_usage"]["sdk_tokens"] / performance_results["sdk_calls"] if performance_results["sdk_calls"] > 0 else 0
    avg_cache_tokens = total_cache_tokens / performance_results["cache_hits"] if performance_results["cache_hits"] > 0 else 0
    tokens_saved = (avg_sdk_tokens * performance_results["cache_hits"]) - total_cache_tokens
    cost_saved = tokens_saved / 1000 * 0.001  # $0.001 per 1K tokens
    token_reduction_pct = ((avg_sdk_tokens - avg_cache_tokens)/avg_sdk_tokens)*100 if avg_sdk_tokens > 0 else 0
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Bar chart comparing average tokens per query
    categories = ['SDK Query', 'Cache Query']
    token_values = [avg_sdk_tokens, avg_cache_tokens]
    
    bars = ax1.bar(categories, token_values, color=['#FF9800', '#2196F3'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Add labels and title
    ax1.set_ylabel('Average Tokens per Query', fontsize=12, fontweight='bold')
    ax1.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add reduction percentage
    ax1.annotate(f"{token_reduction_pct:.1f}% reduction", 
                xy=(1, avg_cache_tokens), 
                xytext=(1.2, (avg_sdk_tokens + avg_cache_tokens)/2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12, fontweight='bold')
    
    # Second subplot: Cost analysis
    # Create a horizontal bar for total savings
    savings_categories = ['Tokens Saved', 'Cost Saved ($)']
    savings_values = [tokens_saved, cost_saved * 1000]  # Multiply by 1000 to make it visible on the same scale
    
    # Use different colors for the different metrics
    colors = ['#4CAF50', '#673AB7']
    
    bars = ax2.barh(savings_categories, savings_values, color=colors)
    
    # Add value labels with appropriate formatting
    ax2.text(savings_values[0] + 100, 0, f'{int(tokens_saved):,} tokens', va='center', fontweight='bold')
    ax2.text(savings_values[1] + 100, 1, f'${cost_saved:.4f}', va='center', fontweight='bold')
    
    # Create a secondary y-axis to show the cost in its natural scale
    ax2_secondary = ax2.twinx()
    ax2_secondary.set_yticks([1])
    ax2_secondary.set_yticklabels([''])
    
    # Add a title
    ax2.set_title('Estimated Savings', fontsize=14, fontweight='bold')
    
    # Set the x-axis label for the primary axis (tokens)
    ax2.set_xlabel('Number of Tokens / Cost ($) × 1000', fontsize=12, fontweight='bold')
    
    # Add a grid for better readability
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add explanatory note
    fig.text(0.5, 0.01, 'Note: Cost calculation based on $0.001 per 1K tokens', 
             ha='center', fontsize=10, style='italic')
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/cost_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/cost_analysis.png"


def visualize_performance_dashboard(performance_results, cache):
    """Create a comprehensive dashboard with all visualizations."""
    # Set up the matplotlib figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(20, 22))
    
    # Adjust the grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Define the axes for each visualization
    ax1 = fig.add_subplot(gs[0, 0])  # Token Usage
    ax2 = fig.add_subplot(gs[0, 1])  # Response Times
    ax3 = fig.add_subplot(gs[1, 0])  # Question Processing Times
    ax4 = fig.add_subplot(gs[1, 1])  # Cache Effectiveness (Pie Chart)
    ax5 = fig.add_subplot(gs[2, 0])  # Cost Analysis
    ax6 = fig.add_subplot(gs[2, 1])  # Similarity Distribution (if available)
    
    # Add a title to the entire figure
    fig.suptitle('Denodo AI Semantic Cache Performance Dashboard', fontsize=24, fontweight='bold', y=0.98)
    
    # 1. Token Usage Visualization
    # Calculate token metrics
    total_cache_tokens = performance_results["token_usage"]["cache_validation_tokens"] + performance_results["token_usage"]["cache_modification_tokens"]
    avg_sdk_tokens = performance_results["token_usage"]["sdk_tokens"] / performance_results["sdk_calls"] if performance_results["sdk_calls"] > 0 else 0
    tokens_saved = (avg_sdk_tokens * performance_results["cache_hits"]) - total_cache_tokens
    
    # Prepare data for visualization
    token_categories = ['SDK Total', 'Cache Validation', 'Cache Modification', 'Total Cache Usage', 'Estimated Savings']
    token_values = [
        performance_results["token_usage"]["sdk_tokens"],
        performance_results["token_usage"]["cache_validation_tokens"],
        performance_results["token_usage"]["cache_modification_tokens"],
        total_cache_tokens,
        tokens_saved if tokens_saved > 0 else 0
    ]
    
    # Create horizontal bar chart for tokens
    bars = ax1.barh(token_categories, token_values, color=sns.color_palette("viridis", len(token_categories)))
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax1.text(label_x_pos + 100, bar.get_y() + bar.get_height()/2, f'{int(width):,}', 
                 va='center', fontweight='bold')
    
    # Add labels and title
    ax1.set_xlabel('Number of Tokens', fontsize=12, fontweight='bold')
    ax1.set_title('Token Usage Comparison', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Response Time Visualization
    # Calculate average times
    avg_cache_time = sum(performance_results["cache_response_times"]) / len(performance_results["cache_response_times"]) if performance_results["cache_response_times"] else 0
    avg_sdk_time = sum(performance_results["sdk_response_times"]) / len(performance_results["sdk_response_times"]) if performance_results["sdk_response_times"] else 0
    time_saved = avg_sdk_time * performance_results["cache_hits"] - avg_cache_time * performance_results["cache_hits"]
    speedup = avg_sdk_time/avg_cache_time if avg_cache_time > 0 else 0
    
    # Prepare data for visualization
    time_categories = ['Average SDK Response', 'Average Cache Response', 'Total Time Saved']
    time_values = [avg_sdk_time, avg_cache_time, time_saved if time_saved > 0 else 0]
    
    # Create horizontal bar chart for response times
    bars = ax2.barh(time_categories, time_values, color=sns.color_palette("cool", len(time_categories)))
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax2.text(label_x_pos + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', 
                 va='center', fontweight='bold')
    
    # Add labels and title
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Response Time Comparison', fontsize=16, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add speedup annotation
    ax2.annotate(f'Cache is {speedup:.1f}x faster than SDK calls', 
                 xy=(0.5, -0.2), xycoords='axes fraction',
                 ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 3. Question Processing Times
    # Get the data
    question_numbers = [f"Q{i+1}" for i in range(len(performance_results["question_texts"]))]
    times = performance_results["all_response_times"]
    sources = performance_results["question_sources"]
    
    # Create color mapping for sources
    colors = {'cache': 'forestgreen', 'sdk': 'orangered'}
    bar_colors = [colors[source] for source in sources]
    
    # Create horizontal bar chart
    bars = ax3.barh(question_numbers, times, color=bar_colors)
    
    # Add value labels and source indicators
    for i, (bar, source) in enumerate(zip(bars, sources)):
        width = bar.get_width()
        source_label = "CACHE" if source == 'cache' else "SDK"
        ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}s ({source_label})', 
                 va='center')
    
    # Add labels and title
    ax3.set_xlabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Question Number', fontsize=12, fontweight='bold')
    ax3.set_title('Question Processing Times', fontsize=16, fontweight='bold')
    
    # Create a custom legend
    legend_elements = [
        Patch(facecolor=colors['cache'], label='Cache'),
        Patch(facecolor=colors['sdk'], label='SDK')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # Add a grid for better readability
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 4. Cache Effectiveness Pie Chart
    # Calculate rates
    cache_hit_rate = (performance_results["cache_hits"] / performance_results["questions_processed"]) * 100 if performance_results["questions_processed"] > 0 else 0
    sdk_call_rate = 100 - cache_hit_rate
    
    # Data for visualization
    labels = ['Cache Hits', 'SDK Calls']
    sizes = [cache_hit_rate, sdk_call_rate]
    
    # Use a vibrant color scheme
    colors = ['#4CAF50', '#FF5722']
    
    # Create a pie chart with a slight explosion for the first slice
    explode = (0.1, 0)  # explode the 1st slice (Cache Hits)
    
    wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=labels, 
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax4.axis('equal')  
    
    # Add a title
    ax4.set_title('Cache Effectiveness', fontsize=16, fontweight='bold')
    
    # Add stats annotation
    stats_text = (f"Total Questions: {performance_results['questions_processed']}\n"
                  f"Cache Hits: {performance_results['cache_hits']}\n"
                  f"SDK Calls: {performance_results['sdk_calls']}")
    
    ax4.annotate(stats_text, xy=(0.5, -0.1), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='center')
    
    # 5. Cost Analysis
    # Calculate token reduction metrics
    total_cache_tokens = performance_results["token_usage"]["cache_validation_tokens"] + performance_results["token_usage"]["cache_modification_tokens"]
    avg_sdk_tokens = performance_results["token_usage"]["sdk_tokens"] / performance_results["sdk_calls"] if performance_results["sdk_calls"] > 0 else 0
    avg_cache_tokens = total_cache_tokens / performance_results["cache_hits"] if performance_results["cache_hits"] > 0 else 0
    token_reduction_pct = ((avg_sdk_tokens - avg_cache_tokens)/avg_sdk_tokens)*100 if avg_sdk_tokens > 0 else 0
    
    # Bar chart comparing average tokens per query
    categories = ['SDK Query', 'Cache Query']
    token_values = [avg_sdk_tokens, avg_cache_tokens]
    
    bars = ax5.bar(categories, token_values, color=['#FF9800', '#2196F3'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Add labels and title
    ax5.set_ylabel('Average Tokens per Query', fontsize=12, fontweight='bold')
    ax5.set_title('Token Usage and Cost Savings', fontsize=16, fontweight='bold')
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add cost information
    tokens_saved = (avg_sdk_tokens * performance_results["cache_hits"]) - total_cache_tokens
    cost_saved = tokens_saved / 1000 * 0.001  # $0.001 per 1K tokens
    
    cost_text = (f"Total Tokens Saved: {int(tokens_saved):,}\n"
                f"Cost Savings: ${cost_saved:.4f}\n"
                f"Token Reduction: {token_reduction_pct:.1f}%")
    
    ax5.annotate(cost_text, xy=(0.5, 0.5), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='center')
    
    # 6. Similarity Distribution (if there are enough cached questions)
    if len(cache.cache["questions"]) >= 2:
        # Calculate similarities between all pairs of questions
        questions = cache.cache["questions"]
        similarities = []
        
        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                # Get embeddings
                embedding_i = cache.cache["embeddings"][i]
                embedding_j = cache.cache["embeddings"][j]
                
                # Calculate cosine similarity (normalized dot product)
                dot_product = np.dot(embedding_i, embedding_j)
                norm_i = np.linalg.norm(embedding_i)
                norm_j = np.linalg.norm(embedding_j)
                similarity = dot_product / (norm_i * norm_j)
                
                similarities.append(similarity)
        
        # Create histogram
        sns.histplot(similarities, bins=20, kde=True, ax=ax6)
        
        # Add a vertical line at the threshold value
        ax6.axvline(x=cache.similarity_threshold, color='r', linestyle='--', 
                    label=f'Threshold ({cache.similarity_threshold})')
        
        # Add labels and title
        ax6.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax6.set_title('Similarity Scores Between Cached Questions', 
                  fontsize=16, fontweight='bold')
        
        # Add legend
        ax6.legend()
        
        # Add grid for better readability
        ax6.grid(linestyle='--', alpha=0.7)
    else:
        # Not enough data for similarity distribution
        ax6.text(0.5, 0.5, 'Not enough cached questions\nto show similarity distribution', 
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax6.transAxes)
        ax6.set_title('Similarity Distribution', fontsize=16, fontweight='bold')
        ax6.axis('off')
    
    # Improve overall layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the overall title
    
    # Save the dashboard
    plt.savefig(f"{VISUALIZATION_OUTPUT_DIR}/performance_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"{VISUALIZATION_OUTPUT_DIR}/performance_dashboard.png"


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
        "question_sources": [],  # 'cache' or 'sdk' for each question
        "question_texts": [],    # store the actual questions
        "all_response_times": [],  # store all response times in order
        "token_usage": {
            "sdk_tokens": 0,
            "cache_validation_tokens": 0,
            "cache_modification_tokens": 0
        }
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
        
        # Store question for reporting
        performance_results["question_texts"].append(question)
        
        # Check if a similar question exists in the cache
        cached_question, cached_sql, cached_result, similarity = cache.find_similar_question(question)
        
        if cached_question:
            print(f"✓ FOUND SIMILAR QUESTION IN CACHE: '{cached_question}'")
            print(f"✓ SIMILARITY SCORE: {similarity:.2f}")
            
            # Use LLM to validate if the questions are truly semantically related
            are_related, explanation = are_questions_semantically_related(cached_question, question)
            
            # Estimate token usage for validation (approximate)
            validation_tokens = len(cached_question + question) / 3
            performance_results["token_usage"]["cache_validation_tokens"] += validation_tokens
            
            if are_related:
                print(f"✓ LLM VALIDATION: Questions are semantically related")
                print(f"✓ EXPLANATION: {explanation}")
                
                # Modify the SQL query for the new question
                start_time = time.time()
                modified_sql = modify_sql_query(cached_sql, question, cached_question)
                
                # Estimate token usage for modification (approximate)
                modification_tokens = (len(cached_question + question + cached_sql) / 3)
                performance_results["token_usage"]["cache_modification_tokens"] += modification_tokens
                
                print(f"\n📜 CACHE MODIFICATION:")
                print(f"Original SQL: {cached_sql}")
                print(f"Modified SQL: {modified_sql}")
                
                # Execute the modified SQL directly using Data Catalog
                status_code, result = execute_vql(modified_sql, auth)
                processing_time = time.time() - start_time
                
                # Update performance metrics
                performance_results["cache_response_times"].append(processing_time)
                performance_results["question_sources"].append("cache")
                performance_results["all_response_times"].append(processing_time)
                performance_results["cache_hits"] += 1
                performance_results["questions_processed"] += 1
                
                if 200 <= status_code < 300:
                    print(f"\n✅ CACHE HIT: SQL execution successful")
                    print(f"⏱️ RESPONSE TIME: {processing_time:.2f} seconds")
                    print(f"💰 ESTIMATED TOKEN SAVINGS: ~3000 tokens")
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
                    performance_results["all_response_times"][-1] = sdk_time  # Update timing
                    
                    if sdk_result:
                        sql_query = sdk_result.get('sql_query', '')
                        query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")
                        
                        # Track token usage
                        token_info = sdk_result.get('tokens', {})
                        total_tokens = token_info.get('total_tokens', 3000)  # Default estimate if not provided
                        performance_results["token_usage"]["sdk_tokens"] += total_tokens

                        print(f"📊 SQL FROM SDK: {sql_query}")
                        print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                        print(f"🔢 TOKENS USED: {total_tokens}")
                        
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
                performance_results["all_response_times"].append(sdk_time)
                
                if sdk_result:
                    sql_query = sdk_result.get('sql_query', '')
                    query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")
                    
                    # Track token usage
                    token_info = sdk_result.get('tokens', {})
                    total_tokens = token_info.get('total_tokens', 3000)  # Default estimate if not provided
                    performance_results["token_usage"]["sdk_tokens"] += total_tokens
                    
                    print(f"📊 SQL FROM SDK: {sql_query}")
                    print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                    print(f"🔢 TOKENS USED: {total_tokens}")
                    
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
            performance_results["all_response_times"].append(sdk_time)
            
            if sdk_result:
                sql_query = sdk_result.get('sql_query', '')
                query_explanation = sdk_result.get('query_explanation', f"Query for: {question}")
                
                # Track token usage
                token_info = sdk_result.get('tokens', {})
                total_tokens = token_info.get('total_tokens', 3000)  # Default estimate if not provided
                performance_results["token_usage"]["sdk_tokens"] += total_tokens
                
                print(f"📊 SQL FROM SDK: {sql_query}")
                print(f"⏱️ SDK RESPONSE TIME: {sdk_time:.2f} seconds")
                print(f"🔢 TOKENS USED: {total_tokens}")
                
                # Add to cache
                cache.add_to_cache(question, sql_query, query_explanation)
            else:
                print("❌ FAILED: No response from Denodo AI SDK")
    
    # Generate all visualizations
    print("\n\n" + "="*80)
    print("           GENERATING PERFORMANCE VISUALIZATIONS           ")
    print("="*80)
    
    # Generate individual visualizations
    token_usage_chart = visualize_token_usage(performance_results)
    print(f"✅ Generated token usage chart: {token_usage_chart}")
    
    response_times_chart = visualize_response_times(performance_results)
    print(f"✅ Generated response times chart: {response_times_chart}")
    
    question_times_chart = visualize_question_processing_times(performance_results)
    print(f"✅ Generated question processing times chart: {question_times_chart}")
    
    cache_effectiveness_chart = visualize_cache_effectiveness(performance_results)
    print(f"✅ Generated cache effectiveness chart: {cache_effectiveness_chart}")
    
    cost_analysis_chart = visualize_cost_analysis(performance_results)
    print(f"✅ Generated cost analysis chart: {cost_analysis_chart}")
    
    similarity_chart = visualize_similarity_distribution(cache)
    if similarity_chart:
        print(f"✅ Generated similarity distribution chart: {similarity_chart}")
    else:
        print("ℹ️ Not enough data for similarity distribution chart")
    
    # Generate comprehensive dashboard
    dashboard = visualize_performance_dashboard(performance_results, cache)
    print(f"✅ Generated comprehensive performance dashboard: {dashboard}")
    
    # Calculate aggregate statistics
    avg_cache_time = sum(performance_results["cache_response_times"]) / len(performance_results["cache_response_times"]) if performance_results["cache_response_times"] else 0
    avg_sdk_time = sum(performance_results["sdk_response_times"]) / len(performance_results["sdk_response_times"]) if performance_results["sdk_response_times"] else 0
    time_saved = avg_sdk_time * performance_results["cache_hits"] - avg_cache_time * performance_results["cache_hits"]
    speedup = avg_sdk_time/avg_cache_time if avg_cache_time > 0 else 0
    
    # Token usage stats
    total_cache_tokens = performance_results["token_usage"]["cache_validation_tokens"] + performance_results["token_usage"]["cache_modification_tokens"]
    avg_sdk_tokens = performance_results["token_usage"]["sdk_tokens"] / performance_results["sdk_calls"] if performance_results["sdk_calls"] > 0 else 0
    avg_cache_tokens = total_cache_tokens / performance_results["cache_hits"] if performance_results["cache_hits"] > 0 else 0
    tokens_saved = (avg_sdk_tokens * performance_results["cache_hits"]) - total_cache_tokens
    
    # Print summary information
    print("\n\n" + "="*80)
    print("                  SEMANTIC CACHE PERFORMANCE REPORT                  ")
    print("="*80)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"{'Total Questions Processed:':<40} {performance_results['questions_processed']}")
    print(f"{'Cache Hits:':<40} {performance_results['cache_hits']} ({(performance_results['cache_hits']/performance_results['questions_processed'])*100:.1f}%)")
    print(f"{'SDK Calls Required:':<40} {performance_results['sdk_calls']}")
    print(f"{'SDK Calls Avoided:':<40} {performance_results['cache_hits']}")
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"{'Estimated Token Savings:':<40} {tokens_saved:,.0f} tokens")
    print(f"{'Estimated Cost Savings:':<40} ${tokens_saved/1000 * 0.001:.4f} (@ $0.001 per 1K tokens)")
    print(f"{'Token Reduction:':<40} {((avg_sdk_tokens - avg_cache_tokens)/avg_sdk_tokens)*100:.1f}% per query")
    
    print(f"\n⏱️ TIME ANALYSIS:")
    print(f"{'Time Savings:':<40} {time_saved:.2f} seconds total")
    print(f"{'Speed Improvement:':<40} {speedup:.1f}x faster with cache")
    
    print(f"\n🖼️ VISUALIZATIONS:")
    print(f"{'Performance Dashboard:':<40} {dashboard}")
    print(f"{'Individual Charts:':<40} {VISUALIZATION_OUTPUT_DIR}/")
    
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
