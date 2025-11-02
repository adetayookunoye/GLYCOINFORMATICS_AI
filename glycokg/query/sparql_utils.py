"""
SPARQL Query Utilities for Glycoinformatics Knowledge Graph

This module provides utilities for executing and managing SPARQL queries
against the glycoinformatics knowledge graph.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

# Check for rdflib availability
try:
    from rdflib import Graph, plugins
    from rdflib.plugins.sparql import prepareQuery
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    
# Check for SPARQLWrapper availability  
try:
    from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, POST, GET
    HAS_SPARQLWRAPPER = True
except ImportError:
    HAS_SPARQLWRAPPER = False
    
# For backward compatibility
HAS_SPARQL = HAS_RDFLIB

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Container for SPARQL query results with metadata."""
    results: List[Dict[str, Any]]
    query: str
    execution_time: float
    result_count: int
    timestamp: datetime
    endpoint: Optional[str] = None
    error: Optional[str] = None


class SPARQLQueryManager:
    """
    Manager for SPARQL queries against glycoinformatics knowledge graphs.
    
    Supports both local RDF graphs and remote SPARQL endpoints with
    query validation, caching, and result formatting.
    """
    
    def __init__(self,
                 endpoint_url: Optional[str] = None,
                 local_graph: Optional[Graph] = None,
                 query_cache_size: int = 100):
        """
        Initialize SPARQL query manager.
        
        Args:
            endpoint_url: URL of remote SPARQL endpoint
            local_graph: Local RDF graph instance
            query_cache_size: Maximum number of cached query results
        """
        self.endpoint_url = endpoint_url
        self.local_graph = local_graph
        self.query_cache: Dict[str, QueryResult] = {}
        self.max_cache_size = query_cache_size
        
        # Initialize SPARQL wrapper for remote endpoint
        if endpoint_url and HAS_SPARQLWRAPPER:
            self.sparql = SPARQLWrapper(endpoint_url)
            self.sparql.setReturnFormat(JSON)
        else:
            self.sparql = None
            
        # Load predefined queries
        self.predefined_queries = self._load_predefined_queries()
        
        logger.info(f"Initialized SPARQL manager with {'remote' if endpoint_url else 'local'} graph")
        
    def _load_predefined_queries(self) -> Dict[str, str]:
        """Load predefined SPARQL queries from files."""
        queries = {}
        
        # Get query directory path
        current_dir = Path(__file__).parent
        query_dir = current_dir / "examples"
        
        if not query_dir.exists():
            logger.warning(f"Query directory not found: {query_dir}")
            return queries
            
        # Load all .sparql files
        for query_file in query_dir.glob("*.sparql"):
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Remove comments and empty lines for key generation
                    lines = [line.strip() for line in content.split('\n') 
                            if line.strip() and not line.strip().startswith('#')]
                    if lines:
                        queries[query_file.stem] = content
                        logger.debug(f"Loaded query: {query_file.stem}")
            except Exception as e:
                logger.warning(f"Failed to load query {query_file}: {e}")
                
        logger.info(f"Loaded {len(queries)} predefined queries")
        return queries
        
    def execute_query(self,
                     query: str,
                     use_cache: bool = True,
                     timeout: int = 30,
                     **parameters) -> QueryResult:
        """
        Execute SPARQL query against the knowledge graph.
        
        Args:
            query: SPARQL query string or predefined query name
            use_cache: Whether to use cached results
            timeout: Query timeout in seconds
            **parameters: Query parameters for substitution
            
        Returns:
            QueryResult with query results and metadata
        """
        start_time = datetime.now()
        
        # Check if it's a predefined query name
        if query in self.predefined_queries:
            query_text = self.predefined_queries[query]
        else:
            query_text = query
            
        # Substitute parameters
        if parameters:
            query_text = self._substitute_parameters(query_text, parameters)
            
        # Check cache
        cache_key = self._generate_cache_key(query_text)
        if use_cache and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            logger.debug(f"Returning cached result for query (age: {datetime.now() - cached_result.timestamp})")
            return cached_result
            
        # Execute query
        try:
            if self.local_graph:
                if not HAS_RDFLIB:
                    raise RuntimeError("rdflib not available for local query execution")
                results = self._execute_local_query(query_text, timeout)
            elif self.sparql:
                results = self._execute_remote_query(query_text, timeout)
            else:
                raise RuntimeError("No graph or SPARQL endpoint configured")
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = QueryResult(
                results=results,
                query=query_text,
                execution_time=execution_time,
                result_count=len(results),
                timestamp=datetime.now(),
                endpoint=self.endpoint_url
            )
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, result)
                
            logger.info(f"Query executed successfully: {len(results)} results in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = f"Query execution failed: {e}"
            logger.error(error_msg)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                results=[],
                query=query_text,
                execution_time=execution_time,
                result_count=0,
                timestamp=datetime.now(),
                endpoint=self.endpoint_url,
                error=error_msg
            )
            
    def _execute_local_query(self, query: str, timeout: int) -> List[Dict[str, Any]]:
        """Execute query against local RDF graph."""
        if not self.local_graph:
            raise RuntimeError("No local graph available")
            
        try:
            # Prepare and execute query
            prepared_query = prepareQuery(query)
            results = self.local_graph.query(prepared_query)
            
            # Convert results to list of dictionaries
            formatted_results = []
            for row in results:
                result_dict = {}
                for var in results.vars:
                    value = row[var] if var in row else None
                    if value is not None:
                        result_dict[str(var)] = str(value)
                formatted_results.append(result_dict)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Local query execution failed: {e}")
            raise
            
    def _execute_remote_query(self, query: str, timeout: int) -> List[Dict[str, Any]]:
        """Execute query against remote SPARQL endpoint."""
        if not self.sparql:
            raise RuntimeError("No SPARQL endpoint configured")
            
        try:
            self.sparql.setQuery(query)
            self.sparql.setTimeout(timeout)
            
            result = self.sparql.query().convert()
            
            # Extract results from SPARQL JSON format
            if isinstance(result, dict) and "results" in result:
                bindings = result["results"]["bindings"]
                formatted_results = []
                
                for binding in bindings:
                    result_dict = {}
                    for var, value_info in binding.items():
                        result_dict[var] = value_info.get("value", "")
                    formatted_results.append(result_dict)
                    
                return formatted_results
            else:
                logger.warning("Unexpected result format from SPARQL endpoint")
                return []
                
        except Exception as e:
            logger.error(f"Remote query execution failed: {e}")
            raise
            
    def _substitute_parameters(self, query: str, parameters: Dict[str, Any]) -> str:
        """Substitute parameters in query template."""
        substituted_query = query
        
        for param, value in parameters.items():
            placeholder = f"${{{param}}}"
            if isinstance(value, str):
                # Add quotes for string literals
                substituted_value = f'"{value}"'
            else:
                substituted_value = str(value)
                
            substituted_query = substituted_query.replace(placeholder, substituted_value)
            
        return substituted_query
        
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        return hashlib.md5(query.encode('utf-8')).hexdigest()
        
    def _cache_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result with size management."""
        # Remove oldest entries if cache is full
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.query_cache.keys(),
                           key=lambda k: self.query_cache[k].timestamp)
            del self.query_cache[oldest_key]
            
        self.query_cache[cache_key] = result
        
    def get_predefined_queries(self) -> List[str]:
        """Get list of available predefined query names."""
        return list(self.predefined_queries.keys())
        
    def get_query_template(self, query_name: str) -> Optional[str]:
        """Get template for a predefined query."""
        return self.predefined_queries.get(query_name)
        
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate SPARQL query syntax.
        
        Args:
            query: SPARQL query string
            
        Returns:
            Validation result with status and error details
        """
        try:
            if HAS_RDFLIB:
                # Try to prepare the query to check syntax
                prepareQuery(query)
                return {
                    "valid": True,
                    "error": None,
                    "message": "Query syntax is valid"
                }
            else:
                return {
                    "valid": False,
                    "error": "SPARQL validation not available",
                    "message": "rdflib not installed"
                }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "message": f"Query syntax error: {e}"
            }
            
    def format_results(self, 
                      results: QueryResult,
                      format_type: str = "table") -> Dict[str, Any]:
        """
        Format query results for different output types.
        
        Args:
            results: QueryResult object
            format_type: Output format (table, json, csv, markdown)
            
        Returns:
            Formatted results with metadata
        """
        formatted = {
            "query": results.query,
            "execution_time": results.execution_time,
            "result_count": results.result_count,
            "timestamp": results.timestamp.isoformat(),
            "data": None,
            "format": format_type
        }
        
        if results.error:
            formatted["error"] = results.error
            return formatted
            
        if format_type == "json":
            formatted["data"] = results.results
            
        elif format_type == "table" or format_type == "markdown":
            if results.results:
                # Create table structure
                headers = list(results.results[0].keys()) if results.results else []
                rows = []
                for result in results.results:
                    row = [result.get(header, "") for header in headers]
                    rows.append(row)
                    
                if format_type == "table":
                    formatted["data"] = {
                        "headers": headers,
                        "rows": rows
                    }
                else:  # markdown
                    # Generate markdown table
                    if headers and rows:
                        md_lines = []
                        md_lines.append("| " + " | ".join(headers) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for row in rows:
                            md_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                        formatted["data"] = "\n".join(md_lines)
                    else:
                        formatted["data"] = "No results found."
                        
        elif format_type == "csv":
            if results.results:
                import io
                import csv
                
                output = io.StringIO()
                headers = list(results.results[0].keys()) if results.results else []
                writer = csv.DictWriter(output, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results.results)
                formatted["data"] = output.getvalue()
            else:
                formatted["data"] = ""
                
        return formatted
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about query manager usage."""
        stats = {
            "endpoint_url": self.endpoint_url,
            "local_graph_size": len(self.local_graph) if self.local_graph else 0,
            "cached_queries": len(self.query_cache),
            "predefined_queries": len(self.predefined_queries),
            "cache_utilization": f"{len(self.query_cache)}/{self.max_cache_size}",
            "available_queries": list(self.predefined_queries.keys())
        }
        
        return stats
        
    def clear_cache(self) -> int:
        """Clear query result cache and return number of cleared entries."""
        cleared_count = len(self.query_cache)
        self.query_cache.clear()
        logger.info(f"Cleared {cleared_count} cached query results")
        return cleared_count


def create_query_manager(config: Optional[Dict[str, Any]] = None) -> SPARQLQueryManager:
    """
    Factory function to create configured SPARQL query manager.
    
    Args:
        config: Configuration dictionary with endpoint settings
        
    Returns:
        Configured SPARQLQueryManager instance
    """
    if config is None:
        config = {}
        
    endpoint_url = config.get("sparql_endpoint")
    local_graph = config.get("local_graph")
    cache_size = config.get("cache_size", 100)
    
    return SPARQLQueryManager(
        endpoint_url=endpoint_url,
        local_graph=local_graph,
        query_cache_size=cache_size
    )


# Example predefined query templates with parameters
QUERY_TEMPLATES = {
    "glycan_by_mass_range": """
        PREFIX glyco: <https://glycokg.org/ontology#>
        SELECT ?glycan_id ?mass ?wurcs WHERE {
            ?glycan a glyco:Oligosaccharide ;
                    glyco:hasGlyTouCanID ?glycan_id ;
                    glyco:hasMonoisotopicMass ?mass .
            FILTER(?mass >= ${min_mass} && ?mass <= ${max_mass})
            OPTIONAL { ?glycan glyco:hasWURCSSequence ?wurcs }
        }
        ORDER BY ?mass
        LIMIT ${limit}
    """,
    
    "protein_glycans_by_organism": """
        PREFIX glyco: <https://glycokg.org/ontology#>
        PREFIX taxonomy: <http://purl.uniprot.org/taxonomy/>
        SELECT ?protein_id ?glycan_id ?confidence WHERE {
            ?association a glyco:ProteinGlycanAssociation ;
                        glyco:hasProtein ?protein ;
                        glyco:hasGlycan ?glycan ;
                        glyco:foundIn taxonomy:${taxonomy_id} ;
                        glyco:hasConfidenceScore ?confidence .
            ?protein glyco:hasUniProtID ?protein_id .
            ?glycan glyco:hasGlyTouCanID ?glycan_id .
            FILTER(?confidence >= ${min_confidence})
        }
        ORDER BY DESC(?confidence)
        LIMIT ${limit}
    """
}


if __name__ == "__main__":
    # Example usage
    manager = SPARQLQueryManager()
    
    print("Available predefined queries:")
    for query_name in manager.get_predefined_queries():
        print(f"  - {query_name}")
        
    # Validate a query
    validation = manager.validate_query("SELECT ?s WHERE { ?s ?p ?o }")
    print(f"\nQuery validation: {validation}")