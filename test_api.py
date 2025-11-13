"""
Simple test script for the CTR Prediction API
Run this while the Flask app is running in another terminal
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*50)
    print("Testing /health endpoint")
    print("="*50)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict_single():
    """Test single ad prediction"""
    print("\n" + "="*50)
    print("Testing /predict_single endpoint")
    print("="*50)
    
    # Sample ad with all 39 raw Criteo features
    sample_data = {
        "features": {
            # 13 integer features
            "I1": 5, "I2": 10, "I3": 2, "I4": 15, "I5": 3,
            "I6": 100, "I7": 50, "I8": 8, "I9": 12, "I10": 25,
            "I11": 7, "I12": 4, "I13": 30,
            # 26 categorical features
            "C1": "1000", "C2": "efba", "C3": "05db", "C4": "fb9c",
            "C5": "25c8", "C6": "7e6e", "C7": "0b15", "C8": "21dd",
            "C9": "a73e", "C10": "b1f8", "C11": "3475", "C12": "8a4f",
            "C13": "e5ba", "C14": "74c2", "C15": "38eb", "C16": "1e88",
            "C17": "3b08", "C18": "7c6e", "C19": "b28b", "C20": "febc",
            "C21": "8f90", "C22": "b04e", "C23": "25c8", "C24": "c9d9",
            "C25": "0014", "C26": "13d5"
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict_single", json=sample_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_score_ad():
    """Test ad scoring endpoint"""
    print("\n" + "="*50)
    print("Testing /score_ad endpoint")
    print("="*50)
    
    # Sample ad with features from the Criteo dataset
    # Using integer and categorical features
    sample_data = {
        "features": {
            "I1": 5,
            "I2": 10,
            "I3": 2,
            "I4": 15,
            "I5": 3,
            "I6": 100,
            "I7": 50,
            "I8": 8,
            "I9": 12,
            "I10": 25,
            "I11": 7,
            "I12": 4,
            "I13": 30,
            "C1": "1000",
            "C2": "efba",
            "C3": "05db",
            "C4": "fb9c",
            "C5": "25c8",
            "C6": "7e6e",
            "C7": "0b15",
            "C8": "21dd",
            "C9": "a73e"
        }
    }
    
    response = requests.post(f"{BASE_URL}/score_ad", json=sample_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CTR PREDICTION API TEST SUITE")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the Flask app is running in another terminal!")
    print("="*60)
    
    try:
        # Test health endpoint
        health_ok = test_health()
        
        # Test scoring endpoint
        score_ok = test_score_ad()
        
        # Test prediction endpoint
        predict_ok = test_predict_single()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Health Check: {'PASSED' if health_ok else 'FAILED'}")
        print(f"✅ Ad Scoring: {'PASSED' if score_ok else 'FAILED'}")
        print(f"✅ Single Prediction: {'PASSED' if predict_ok else 'FAILED'}")
        print("="*60)
        
        if health_ok and score_ok and predict_ok:
            print("\n🎉 ALL TESTS PASSED! API is fully functional!")
        else:
            print("\n⚠️  Some tests failed. Check logs above.")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API")
        print("Make sure the Flask app is running with: python app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
