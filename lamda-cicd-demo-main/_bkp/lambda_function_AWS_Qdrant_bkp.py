import json
import os
import logging
from qdrant_simple import QdrantHTTPClient, Point

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    AWS Lambda handler function for Qdrant operations
    """
    try:
        # Log the incoming event for debugging
        logger.info(f"Received event: {json.dumps(event)}")

        # Get Qdrant connection details from environment variables
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        # Validate environment variables qdrant_url and qdrant_api_key
        if not qdrant_url or not qdrant_api_key:
            error_response = {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": "Missing QDRANT_URL or QDRANT_API_KEY environment variables",
                        "qdrant_url_set": bool(qdrant_url),
                        "qdrant_api_key_set": bool(qdrant_api_key),
                    }
                ),
            }
            logger.error(f"Environment validation failed: {error_response}")
            return error_response

        # Initialize Qdrant HTTP client
        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantHTTPClient(
            url=qdrant_url, api_key=qdrant_api_key, timeout=30
        )

        # Parse the operation from event
        operation = event.get("operation", "health_check")
        logger.info(f"Processing operation: {operation}")

        if operation == "health_check":
            return health_check(qdrant_client)
        elif operation == "create_collection":
            return create_collection(qdrant_client, event)
        elif operation == "insert_points":
            return insert_points(qdrant_client, event)
        elif operation == "search":
            return search_points(qdrant_client, event)
        elif operation == "list_collections":
            return list_collections(qdrant_client)
        elif operation == "collection_info":
            return collection_info(qdrant_client, event)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": f"Unknown operation: {operation}",
                        "available_operations": [
                            "health_check",
                            "create_collection",
                            "insert_points",
                            "search",
                            "list_collections",
                            "collection_info",
                        ],
                    }
                ),
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Invalid JSON in request: {str(e)}"}),
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }


def health_check(client):
    """Check Qdrant connection health"""
    try:
        logger.info("Performing health check...")
        # Get cluster info to verify connection
        cluster_info = client.get_cluster_info()
        logger.info("Health check successful")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "healthy",
                    "cluster_info": cluster_info,
                    "message": "Qdrant connection is working properly",
                }
            ),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "unhealthy", "error": str(e)}),
        }


def create_collection(client, event):
    """Create a new collection"""
    try:
        collection_name = event.get("collection_name", "test_collection")
        vector_size = event.get("vector_size", 384)
        distance_metric = event.get("distance", "Cosine")

        logger.info(f"Creating collection: {collection_name}")

        # Check if collection already exists
        if client.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": f"Collection {collection_name} already exists",
                        "collection_name": collection_name,
                        "status": "exists",
                    }
                ),
            }

        # Create collection
        result = client.create_collection(
            name=collection_name, vector_size=vector_size, distance=distance_metric
        )

        logger.info(f"Collection {collection_name} created successfully")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Collection {collection_name} created successfully",
                    "collection_name": collection_name,
                    "vector_size": vector_size,
                    "distance": distance_metric,
                    "result": result,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to create collection: {str(e)}"}),
        }


def insert_points(client, event):
    """Insert points into a collection"""
    try:
        collection_name = event.get("collection_name", "test_collection")
        points_data = event.get("points", [])

        if not points_data:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No points data provided"}),
            }

        logger.info(f"Inserting {len(points_data)} points into {collection_name}")

        # Convert points data to Point objects
        points = []
        for point_data in points_data:
            point = Point(
                id=point_data.get("id"),
                vector=point_data.get("vector", []),
                payload=point_data.get("payload", {}),
            )
            points.append(point)

        # Insert points
        result = client.upsert_points(collection_name=collection_name, points=points)

        logger.info(f"Successfully inserted {len(points)} points")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Inserted {len(points)} points successfully",
                    "collection_name": collection_name,
                    "points_count": len(points),
                    "result": result,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to insert points: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to insert points: {str(e)}"}),
        }


def search_points(client, event):
    """Search for similar points"""
    try:
        collection_name = event.get("collection_name", "test_collection")
        query_vector = event.get("query_vector", [])
        limit = event.get("limit", 5)
        score_threshold = event.get("score_threshold", 0.0)

        if not query_vector:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No query vector provided"}),
            }

        logger.info(f"Searching in collection {collection_name} with limit {limit}")

        # Perform search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Format results
        results = []
        for result in search_results:
            results.append(
                {"id": result.id, "score": result.score, "payload": result.payload}
            )

        logger.info(f"Found {len(results)} search results")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "results": results,
                    "count": len(results),
                    "collection_name": collection_name,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to search points: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to search points: {str(e)}"}),
        }


def list_collections(client):
    """List all collections"""
    try:
        logger.info("Listing all collections")
        collections = client.list_collections()

        logger.info(f"Found {len(collections)} collections")
        return {
            "statusCode": 200,
            "body": json.dumps({"collections": collections, "count": len(collections)}),
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to list collections: {str(e)}"}),
        }


def collection_info(client, event):
    """Get collection information"""
    try:
        collection_name = event.get("collection_name", "test_collection")

        logger.info(f"Getting info for collection: {collection_name}")
        info = client.collection_info(collection_name)

        return {
            "statusCode": 200,
            "body": json.dumps({"collection_name": collection_name, "info": info}),
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to get collection info: {str(e)}"}),
        }
