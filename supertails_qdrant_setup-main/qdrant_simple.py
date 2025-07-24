"""
Minimal HTTP-only Qdrant client to avoid large dependencies
"""

import httpx
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Point:
    id: int
    vector: List[float]
    payload: Dict[str, Any] = None


@dataclass
class SearchResult:
    id: int
    score: float
    payload: Dict[str, Any]


class QdrantHTTPClient:
    def __init__(self, url: str, api_key: str, timeout: int = 300):
        self.base_url = url.rstrip("/")
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.timeout = timeout

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Qdrant API"""
        url = f"{self.base_url}{endpoint}"

        with httpx.Client(timeout=self.timeout) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=self.headers)
            elif method.upper() == "POST":
                response = client.post(url, headers=self.headers, json=data)
            elif method.upper() == "PUT":
                response = client.put(url, headers=self.headers, json=data)
            elif method.upper() == "DELETE":
                response = client.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json() if response.content else {}

    def get_cluster_info(self) -> Dict:
        """Get cluster information"""
        return self._make_request("GET", "/cluster")

    def list_collections(self) -> List[str]:
        """List all collections"""
        response = self._make_request("GET", "/collections")
        return [
            col["name"] for col in response.get("result", {}).get("collections", [])
        ]

    def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> Dict:
        """Create a new collection"""
        data = {"vectors": {"size": vector_size, "distance": distance}}
        return self._make_request("PUT", f"/collections/{name}", data)

    def collection_exists(self, name: str) -> bool:
        """Check if collection exists"""
        try:
            self._make_request("GET", f"/collections/{name}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise

    def upsert_points(self, collection_name: str, points: List[Point]) -> Dict:
        """Insert or update points in collection"""
        data = {
            "points": [
                {"id": point.id, "vector": point.vector, "payload": point.payload or {}}
                for point in points
            ]
        }
        return self._make_request("PUT", f"/collections/{collection_name}/points", data)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        data = {
            "vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
            "with_payload": True,
        }

        response = self._make_request(
            "POST", f"/collections/{collection_name}/points/search", data
        )

        results = []
        for item in response.get("result", []):
            results.append(
                SearchResult(
                    id=item["id"], score=item["score"], payload=item.get("payload", {})
                )
            )

        return results

    def get_point(self, collection_name: str, point_id: int) -> Dict:
        """Get a specific point by ID"""
        return self._make_request(
            "GET", f"/collections/{collection_name}/points/{point_id}"
        )

    def delete_points(self, collection_name: str, point_ids: List[int]) -> Dict:
        """Delete points from collection"""
        data = {"points": point_ids}
        return self._make_request(
            "POST", f"/collections/{collection_name}/points/delete", data
        )

    def collection_info(self, collection_name: str) -> Dict:
        """Get collection information"""
        return self._make_request("GET", f"/collections/{collection_name}")
