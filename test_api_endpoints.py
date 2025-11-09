#!/usr/bin/env python3
"""
Test API Endpoints for Glycomics Databases
Let's verify the correct endpoints and API structure
"""

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import json

def test_glytoucan_sparql():
    """Test GlyTouCan SPARQL endpoint with a simple query"""
    print("🧪 Testing GlyTouCan SPARQL...")
    
    sparql_endpoint = "https://ts.glytoucan.org/sparql"
    sparql = SPARQLWrapper(sparql_endpoint)
    sparql.setReturnFormat(JSON)
    
    # Simple test query
    query = """
    PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
    PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
    
    SELECT DISTINCT ?id
    WHERE {
        ?uri a glycan:saccharide ;
             glytoucan:has_primary_id ?id .
    }
    LIMIT 5
    """
    
    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        print(f"✅ SPARQL Success: Found {len(results['results']['bindings'])} results")
        for result in results["results"]["bindings"]:
            print(f"  - {result['id']['value']}")
        return True
    except Exception as e:
        print(f"❌ SPARQL Failed: {e}")
        return False

def test_glytoucan_rest():
    """Test various GlyTouCan REST API endpoints"""
    print("\n🧪 Testing GlyTouCan REST API...")
    
    test_urls = [
        "https://api.glytoucan.org/glycan/G00047MO",
        "https://glytoucan.org/api/glycan/G00047MO", 
        "https://glycosmos.org/glycans/G00047MO",
        "https://api.glycosmos.org/glytoucan/G00047MO"
    ]
    
    for url in test_urls:
        try:
            print(f"Testing: {url}")
            response = requests.get(url, timeout=10)
            print(f"  Status: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            if response.status_code == 200:
                if 'application/json' in response.headers.get('content-type', ''):
                    data = response.json()
                    print(f"  ✅ JSON Response: {type(data)}")
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                else:
                    print(f"  Content preview: {response.text[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
def test_glygen_api():
    """Test GlyGen API endpoints"""
    print("\n🧪 Testing GlyGen API...")
    
    test_urls = [
        "https://api.glygen.org/glycan/detail/G00047MO",
        "https://api.glygen.org/glycan/G00047MO",
        "https://data.glygen.org/api/glycan/detail/G00047MO"
    ]
    
    for url in test_urls:
        try:
            print(f"Testing: {url}")
            response = requests.get(url, timeout=10)
            print(f"  Status: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            if response.status_code == 200:
                if 'application/json' in response.headers.get('content-type', ''):
                    data = response.json()
                    print(f"  ✅ JSON Response: {type(data)}")
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_pubmed_api():
    """Test PubMed API"""
    print("\n🧪 Testing PubMed API...")
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": "glycan structure",
        "retmax": 3,
        "retmode": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PubMed Success: {data.get('esearchresult', {}).get('count', 0)} papers found")
            pmids = data.get('esearchresult', {}).get('idlist', [])
            print(f"Sample PMIDs: {pmids}")
            
    except Exception as e:
        print(f"❌ PubMed Error: {e}")

def main():
    """Run all API tests"""
    print("🔬 Testing Glycomics Database APIs")
    print("="*50)
    
    test_glytoucan_sparql()
    test_glytoucan_rest()
    test_glygen_api()
    test_pubmed_api()
    
    print("\n✅ API Testing Complete!")

if __name__ == "__main__":
    main()