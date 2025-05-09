"""
RAG system for general hotel information.
Handles vector storage and retrieval for facility and general information queries.
"""

from typing import List, Dict, Any
import boto3
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
from utils.logger import logger
import openai

service = "es"
region = settings.REGION
credentials = boto3.Session().get_credentials()
creds = credentials.get_frozen_credentials()

print(f"Access Key: {creds.access_key}")
print(f"Secret Key: {creds.secret_key}")
print(f"Session Token: {creds.token}")
auth = AWSV4SignerAuth(credentials, region, service)

class GeneralInfoRAG:
    def __init__(self):
        print("creds:{}".format({'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}))
        self.opensearch_client = OpenSearch(
            hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=5,
            retry_on_timeout=True,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def query(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Query the vector store for relevant general information"""
        try:
            # Get embedding for the query
            response = openai.embeddings.create(
                input=query,
                model=settings.EMBEDDING_MODEL
            )
            query_embedding = response.data[0].embedding
            
            # KNN query
            knn_query = {
                "size": k,
                "_source":  ["id", "question", "answer"],
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                }
            }
            
            # Execute KNN search
            response = self.opensearch_client.search(
                index="genie-cassacook-info",
                body=knn_query
            )
            
            if response["hits"]["hits"]:
                return [{
                    "id": hit["_source"].get("id", ""),
                    "question": hit["_source"].get("question", ""),
                    "answer": hit["_source"].get("answer", "")
                } for hit in response["hits"]["hits"]]
            
            # Fallback to similarity search if KNN fails
            logger.warning("KNN search failed, falling back to similarity search")
            return []
            
        except Exception as e:
            logger.error(f"Error querying RAG: {str(e)}")
            return []
        
    def _fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback search using similarity calculator with retry.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - id: Unique identifier
            - question: The question this information addresses
            - answer: The detailed answer or information
        """
        for attempt in range(settings.MAX_RETRIES):
            try:
                # Get all documents from OpenSearch
                response = self.opensearch_client.search(
                    index="genie-hotel-facilities",
                    body={
                        "size": 10,
                        "_source": ["id", "question", "answer"]
                    }
                )
                
                documents = [hit["_source"] for hit in response["hits"]["hits"]]
                if not documents:
                    if attempt < settings.MAX_RETRIES - 1:
                        logger.warning(f"Fallback search attempt {attempt + 1} returned no documents, retrying...")
                        continue
                    return []
                
                # Find most similar documents
                results = []
                for doc in documents:
                    # Calculate similarity using OpenAI embeddings
                    doc_embedding = openai.embeddings.create(
                        input=doc["question"] + " " + doc["answer"],
                        model=settings.EMBEDDING_MODEL
                    ).data[0].embedding
                    
                    query_embedding = openai.embeddings.create(
                        input=query,
                        model=settings.EMBEDDING_MODEL
                    ).data[0].embedding
                    
                    # Calculate cosine similarity
                    similarity = sum(a * b for a, b in zip(doc_embedding, query_embedding)) / (
                        sum(a * a for a in doc_embedding) ** 0.5 * 
                        sum(b * b for b in query_embedding) ** 0.5
                    )
                    
                    if similarity >= settings.SIMILARITY_THRESHOLD:
                        results.append({
                            "id": doc.get("id", ""),
                            "question": doc.get("question", ""),
                            "answer": doc.get("answer", "")
                        })
                    
                    if len(results) >= k:
                        break
                
                if results:
                    return results
                
                if attempt < settings.MAX_RETRIES - 1:
                    logger.warning(f"Fallback search attempt {attempt + 1} found no matches, retrying...")
                    continue
                
                return []
                
            except Exception as e:
                logger.error(f"Error in fallback search attempt {attempt + 1}: {str(e)}")
                if attempt < settings.MAX_RETRIES - 1:
                    continue
                return [] 