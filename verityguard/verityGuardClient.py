"""
VerityGuard API Client
======================
Simple Python client for interacting with the VerityGuard API.

Usage:
    from verityguard_client import VerityGuardClient
    
    client = VerityGuardClient("http://localhost:5001")
    result = client.analyze_post("Breaking news: Amazing discovery!")
    print(result['verdict'])
"""

import requests
import base64
from pathlib import Path
from typing import Optional, Dict, List


class VerityGuardClient:
    """Client for VerityGuard Fake News Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the VerityGuard API
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
    
    def health_check(self) -> Dict:
        """
        Check API health status
        
        Returns:
            Health status information
        """
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
    
    def analyze_post(
        self, 
        text: str, 
        image_path: Optional[str] = None,
        username: str = "anonymous"
    ) -> Dict:
        """
        Analyze a single post for fake news
        
        Args:
            text: Post text content
            image_path: Optional path to image file
            username: Username of post author
        
        Returns:
            Analysis results with verdict, score, and detailed analysis
        """
        files = {}
        data = {
            'text': text,
            'username': username
        }
        
        # Add image if provided
        if image_path and Path(image_path).exists():
            with open(image_path, 'rb') as f:
                files['image'] = f
                response = requests.post(
                    f"{self.api_url}/predict",
                    data=data,
                    files=files
                )
        else:
            response = requests.post(
                f"{self.api_url}/predict",
                data=data
            )
        
        response.raise_for_status()
        return response.json()
    
    def analyze_post_base64(
        self,
        text: str,
        image_base64: Optional[str] = None,
        username: str = "anonymous"
    ) -> Dict:
        """
        Analyze a post with base64-encoded image
        
        Args:
            text: Post text content
            image_base64: Base64-encoded image string
            username: Username of post author
        
        Returns:
            Analysis results
        """
        payload = {
            'text': text,
            'username': username
        }
        
        if image_base64:
            payload['image_base64'] = image_base64
        
        response = requests.post(
            f"{self.api_url}/predict/base64",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_process(self, csv_path: str) -> Dict:
        """
        Process a batch of posts from CSV file
        
        Args:
            csv_path: Path to CSV file with columns: text, image_path, username
        
        Returns:
            Batch processing results with statistics
        """
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.api_url}/batch",
                files=files
            )
        
        response.raise_for_status()
        return response.json()
    
    def download_batch_results(self, batch_id: str, output_path: str):
        """
        Download batch results to file
        
        Args:
            batch_id: Batch ID from batch_process response
            output_path: Where to save the results file
        """
        response = requests.get(
            f"{self.api_url}/batch/{batch_id}/download",
            stream=True
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def get_recent_results(self, limit: int = 100) -> List[Dict]:
        """
        Get recent analysis results
        
        Args:
            limit: Maximum number of results to return
        
        Returns:
            List of recent analysis results
        """
        response = requests.get(
            f"{self.api_url}/results",
            params={'limit': limit}
        )
        response.raise_for_status()
        return response.json()['results']
    
    def get_statistics(self) -> Dict:
        """
        Get overall detection statistics
        
        Returns:
            Statistics including total analyzed, fake count, etc.
        """
        response = requests.get(f"{self.api_url}/stats")
        response.raise_for_status()
        return response.json()['statistics']


# ============================================================
# USAGE EXAMPLES
# ============================================================

def example_single_post():
    """Example: Analyze a single post"""
    client = VerityGuardClient()
    
    # Simple text analysis
    result = client.analyze_post(
        text="BREAKING: Scientists discover cure for all diseases!",
        username="newsbot"
    )
    
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Score: {result['score']:.4f}")


def example_with_image():
    """Example: Analyze post with image"""
    client = VerityGuardClient()
    
    result = client.analyze_post(
        text="Check out this amazing sunset!",
        image_path="sunset.jpg",
        username="photographer"
    )
    
    print(f"Verdict: {result['verdict']}")
    print(f"VAD Valence: {result['detailed_analysis']['vad_analysis']['valence']:.2f}")


def example_batch_processing():
    """Example: Process a batch of posts"""
    client = VerityGuardClient()
    
    # Upload CSV for processing
    result = client.batch_process("posts.csv")
    
    print(f"Processed: {result['statistics']['total']} posts")
    print(f"Fake: {result['statistics']['fake']}")
    print(f"Real: {result['statistics']['real']}")
    
    # Download full results
    client.download_batch_results(
        batch_id=result['batch_id'],
        output_path=f"results_{result['batch_id']}.json"
    )


def example_monitoring():
    """Example: Monitor system statistics"""
    client = VerityGuardClient()
    
    # Check health
    health = client.health_check()
    print(f"System Status: {health['status']}")
    print(f"Device: {health['device']}")
    
    # Get statistics
    stats = client.get_statistics()
    print(f"\nTotal Analyzed: {stats['total_analyzed']}")
    print(f"Fake Rate: {stats['fake_count']/stats['total_analyzed']:.2%}")
    print(f"Average Confidence: {stats['average_confidence']:.2%}")


def example_recent_history():
    """Example: Get recent analysis history"""
    client = VerityGuardClient()
    
    results = client.get_recent_results(limit=10)
    
    print("Recent Analyses:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['post_id']}: {result['verdict']} ({result['confidence']:.2%})")


if __name__ == '__main__':
    print("="*70)
    print("VerityGuard API Client - Examples")
    print("="*70)
    
    # Check if server is running
    try:
        client = VerityGuardClient()
        health = client.health_check()
        print(f"\n✓ Connected to VerityGuard API")
        print(f"  Status: {health['status']}")
        print(f"  Device: {health['device']}")
        
        print("\n" + "="*70)
        print("Running Examples...")
        print("="*70)
        
        # Uncomment to run examples:
        # example_single_post()
        # example_with_image()
        # example_batch_processing()
        # example_monitoring()
        # example_recent_history()
        
        print("\nUncomment example functions to try them!")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to VerityGuard API")
        print("  Make sure the server is running: python app.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")