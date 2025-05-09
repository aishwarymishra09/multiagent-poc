"""
RAG system for dining information.
Handles vector storage and retrieval for dining-related queries.
"""

from typing import List, Dict, Any
import boto3
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
from utils.logger import logger
from utils.similarity import similarity_calculator
import openai

service = "es"
region = settings.REGION
credentials = boto3.Session().get_credentials()
creds = credentials.get_frozen_credentials()

print(f"Access Key: {creds.access_key}")
print(f"Secret Key: {creds.secret_key}")
print(f"Session Token: {creds.token}")
auth = AWSV4SignerAuth(credentials, region, service)

class DiningRAG:
    def __init__(self):

        print("creds:{}".format({'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}))
        self.opensearch_client = OpenSearch(
            hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,  # Increase timeout (default is 10)
            max_retries=5,  # Retry in case of failure
            retry_on_timeout=True,
        )
        
        # self.vectorstore = OpenSearchVectorSearch(
        #     embedding=OpenAIEmbeddings(model=settings.EMBEDDING_MODEL),
        #     opensearch_url=f"https://{settings.OPENSEARCH_HOST}:{settings.OPENSEARCH_PORT}",
        #     index_name="dining",
        #     http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
        #     use_ssl=True,
        #     verify_certs=True,
        #     ssl_assert_hostname=False,
        #     ssl_show_warn=False
        # )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def query(self, query: str, k: int = settings.MAX_RESULTS) -> List[Dict[str, Any]]:
        """Query the vector store for relevant information"""
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
                "_source": ["id", "name", "description", "category", "tags", "dietaryTags", "ingredients", "price_base", "currency"],
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
                index="genie-room-dining-items",
                body=knn_query
            )
            
            if response["hits"]["hits"]:
                return [{
                    "id": hit["_source"].get("id", ""),
                    "name": hit["_source"].get("name", ""),
                    "description": hit["_source"].get("description", ""),
                    "category": hit["_source"].get("category", ""),
                    "tags": hit["_source"].get("tags", []),
                    "dietaryTags": hit["_source"].get("dietaryTags", []),
                    "ingredients": hit["_source"].get("ingredients", []),
                    "price": hit["_source"].get("price_base", 0),
                    "currency": hit["_source"].get("currency", "EURO")
                } for hit in response["hits"]["hits"]]
            
            # Fallback to similarity search if KNN fails
            logger.warning("KNN search failed, falling back to similarity search")
            return self._fallback_search(query, k)
            
        except Exception as e:
            logger.error(f"Error querying RAG: {str(e)}")
            return self._fallback_search(query, k)
    
    # async def add_documents(self, documents: List[Dict[str, Any]]):
    #     """Add new documents to the vector store"""
    #     try:
    #         texts = self.text_splitter.split_documents(documents)
    #         self.vectorstore.add_documents(texts)
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def _fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback search using similarity calculator with retry"""
        for attempt in range(settings.MAX_RETRIES):
            try:
                # Get all documents from OpenSearch
                response = self.opensearch_client.search(
                    index="dining",
                    body={
                        "size": 10,  # Get more documents for better matching
                        "_source": ["id", "name", "description", "category", "tags", "dietaryTags", "ingredients", "price_base", "currency"]
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
                    similarity = similarity_calculator.find_most_similar(
                        query,
                        [doc["description"]]  # Use description for similarity matching
                    )
                    
                    if similarity["similarity"] >= settings.SIMILARITY_THRESHOLD:
                        results.append({
                            "id": doc.get("id", ""),
                            "name": doc.get("name", ""),
                            "description": doc.get("description", ""),
                            "category": doc.get("category", ""),
                            "tags": doc.get("tags", []),
                            "dietaryTags": doc.get("dietaryTags", []),
                            "ingredients": doc.get("ingredients", []),
                            "price": doc.get("price_base", 0),
                            "currency": doc.get("currency", "EURO")
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