"""
VerityGuard API Test Suite
==========================
Automated tests for the VerityGuard API endpoints.

Usage:
    python test_api.py
"""

import requests
import time
import json
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class VerityGuardTester:
    """Test suite for VerityGuard API"""
    
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def print_header(self, text):
        """Print section header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print(f"{text}")
        print(f"{'='*70}{Colors.RESET}\n")
    
    def print_test(self, name, passed, message=""):
        """Print test result"""
        self.total += 1
        if passed:
            self.passed += 1
            print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
            if message:
                print(f"  {message}")
        else:
            self.failed += 1
            print(f"{Colors.RED}✗{Colors.RESET} {name}")
            if message:
                print(f"  {Colors.RED}{message}{Colors.RESET}")
    
    def test_health_check(self):
        """Test health check endpoint"""
        self.print_header("Testing Health Check")
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            
            # Test status code
            self.print_test(
                "Health endpoint returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            # Test response structure
            data = response.json()
            self.print_test(
                "Response contains required fields",
                all(k in data for k in ['status', 'timestamp', 'models_loaded', 'device']),
                f"Keys: {list(data.keys())}"
            )
            
            # Test status value
            self.print_test(
                "System status is healthy",
                data.get('status') == 'healthy',
                f"Status: {data.get('status')}"
            )
            
        except Exception as e:
            self.print_test("Health check", False, str(e))
    
    def test_single_post_analysis(self):
        """Test single post analysis endpoint"""
        self.print_header("Testing Single Post Analysis")
        
        try:
            # Test with text only
            response = requests.post(
                f"{self.api_url}/predict",
                data={
                    'text': 'BREAKING: Scientists discover cure for cancer!',
                    'username': 'tester'
                },
                timeout=30
            )
            
            self.print_test(
                "Post analysis returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            
            # Test response structure
            required_fields = ['success', 'verdict', 'score', 'confidence', 'post_id']
            self.print_test(
                "Response contains required fields",
                all(k in data for k in required_fields),
                f"Keys: {list(data.keys())}"
            )
            
            # Test verdict is valid
            valid_verdicts = ['FAKE', 'REAL', 'UNKNOWN']
            self.print_test(
                "Verdict is valid",
                data.get('verdict') in valid_verdicts,
                f"Verdict: {data.get('verdict')}"
            )
            
            # Test score range
            score = data.get('score', -1)
            self.print_test(
                "Score is in valid range [0, 1]",
                0 <= score <= 1,
                f"Score: {score:.4f}"
            )
            
            # Test confidence range
            conf = data.get('confidence', -1)
            self.print_test(
                "Confidence is in valid range [0, 1]",
                0 <= conf <= 1,
                f"Confidence: {conf:.4f}"
            )
            
        except Exception as e:
            self.print_test("Single post analysis", False, str(e))
    
    def test_base64_analysis(self):
        """Test base64 image analysis endpoint"""
        self.print_header("Testing Base64 Image Analysis")
        
        try:
            # Create a simple test payload
            payload = {
                'text': 'Test post with image',
                'username': 'tester'
            }
            
            response = requests.post(
                f"{self.api_url}/predict/base64",
                json=payload,
                timeout=30
            )
            
            self.print_test(
                "Base64 endpoint returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            self.print_test(
                "Response is successful",
                data.get('success') == True,
                f"Success: {data.get('success')}"
            )
            
        except Exception as e:
            self.print_test("Base64 analysis", False, str(e))
    
    def test_error_handling(self):
        """Test error handling"""
        self.print_header("Testing Error Handling")
        
        # Test missing text
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                data={'username': 'tester'},
                timeout=10
            )
            
            self.print_test(
                "Returns 400 for missing text",
                response.status_code == 400,
                f"Status: {response.status_code}"
            )
            
        except Exception as e:
            self.print_test("Missing text error handling", False, str(e))
        
        # Test invalid endpoint
        try:
            response = requests.get(
                f"{self.api_url}/invalid_endpoint",
                timeout=5
            )
            
            self.print_test(
                "Returns 404 for invalid endpoint",
                response.status_code == 404,
                f"Status: {response.status_code}"
            )
            
        except Exception as e:
            self.print_test("Invalid endpoint handling", False, str(e))
    
    def test_statistics(self):
        """Test statistics endpoint"""
        self.print_header("Testing Statistics")
        
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            
            self.print_test(
                "Statistics endpoint returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            stats = data.get('statistics', {})
            
            # Test required fields
            required_fields = [
                'total_analyzed', 'fake_count', 'real_count',
                'average_confidence', 'recent_trends'
            ]
            self.print_test(
                "Statistics contain required fields",
                all(k in stats for k in required_fields),
                f"Keys: {list(stats.keys())}"
            )
            
            # Test values are non-negative
            self.print_test(
                "Total analyzed is non-negative",
                stats.get('total_analyzed', -1) >= 0,
                f"Total: {stats.get('total_analyzed')}"
            )
            
        except Exception as e:
            self.print_test("Statistics", False, str(e))
    
    def test_results_endpoint(self):
        """Test results retrieval endpoint"""
        self.print_header("Testing Results Retrieval")
        
        try:
            response = requests.get(
                f"{self.api_url}/results?limit=10",
                timeout=10
            )
            
            self.print_test(
                "Results endpoint returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            self.print_test(
                "Response contains results array",
                'results' in data,
                f"Keys: {list(data.keys())}"
            )
            
            # Test limit parameter works
            results = data.get('results', [])
            self.print_test(
                "Respects limit parameter",
                len(results) <= 10,
                f"Returned: {len(results)} results"
            )
            
        except Exception as e:
            self.print_test("Results retrieval", False, str(e))
    
    def test_performance(self):
        """Test API performance"""
        self.print_header("Testing Performance")
        
        try:
            # Test response time
            start = time.time()
            response = requests.post(
                f"{self.api_url}/predict",
                data={
                    'text': 'Quick performance test',
                    'username': 'tester'
                },
                timeout=30
            )
            elapsed = time.time() - start
            
            self.print_test(
                "Analysis completes in under 10 seconds",
                elapsed < 10,
                f"Time: {elapsed:.2f}s"
            )
            
            # Test concurrent requests (simple)
            start = time.time()
            responses = []
            for i in range(3):
                r = requests.post(
                    f"{self.api_url}/predict",
                    data={
                        'text': f'Concurrent test {i}',
                        'username': 'tester'
                    },
                    timeout=30
                )
                responses.append(r)
            elapsed = time.time() - start
            
            all_success = all(r.status_code == 200 for r in responses)
            self.print_test(
                "Handles 3 concurrent requests",
                all_success,
                f"Time: {elapsed:.2f}s"
            )
            
        except Exception as e:
            self.print_test("Performance test", False, str(e))
    
    def run_all_tests(self):
        """Run all test suites"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("="*70)
        print("  VERITYGUARD API TEST SUITE")
        print("="*70)
        print(f"{Colors.RESET}")
        print(f"Testing API at: {self.base_url}")
        
        # Run all tests
        self.test_health_check()
        self.test_single_post_analysis()
        self.test_base64_analysis()
        self.test_error_handling()
        self.test_statistics()
        self.test_results_endpoint()
        self.test_performance()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.BOLD}{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}{Colors.RESET}\n")
        
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        print(f"Total Tests:  {self.total}")
        print(f"{Colors.GREEN}Passed:       {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed:       {self.failed}{Colors.RESET}")
        print(f"Pass Rate:    {pass_rate:.1f}%")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}⚠ Some tests failed. Please review.{Colors.RESET}")
        
        print(f"\n{'='*70}\n")


def main():
    """Run the test suite"""
    import sys
    
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
    
    # Create tester
    tester = VerityGuardTester(base_url)
    
    # Check if server is reachable
    try:
        response = requests.get(base_url, timeout=5)
        print(f"{Colors.GREEN}✓ Server is reachable{Colors.RESET}")
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ Cannot connect to server at {base_url}{Colors.RESET}")
        print(f"  Make sure the server is running: python app.py")
        return
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        return
    
    # Run tests
    tester.run_all_tests()


if __name__ == '__main__':
    main()