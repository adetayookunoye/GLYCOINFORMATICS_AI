from fastapi import FastAPI, Response, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json, yaml, time, uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from rdflib import Graph
from jsonschema import validate, ValidationError
from pathlib import Path

# Platform imports
try:
    from glycokg.integration.coordinator import DataCoordinator
    from glycogot.integration import ReasoningOrchestrator, ReasoningRequest
    from glycogot.reasoning import ReasoningType
    from glycollm.models.glycollm import GlycoLLM
    HAS_PLATFORM_COMPONENTS = True
except ImportError:
    # Fallback for basic operation
    HAS_PLATFORM_COMPONENTS = False
    logging.warning("Platform components not available - running in basic mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_START = time.time()

# Metrics
REQUESTS = Counter("glyco_requests_total", "Total API requests", ["route"])
LATENCY = Histogram("glyco_latency_seconds", "Request latency", ["route"])

# Global platform components
platform_components = {}

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"
CONFIGS = ROOT / "configs"
KG_SAMPLE = ROOT / "glycokg" / "graph" / "sample.ttl"

with open(CONFIGS / "policy.yaml") as f:
    POLICY = yaml.safe_load(f)

with open(SCHEMAS / "spectra_v1.json") as f:
    SPECTRA_SCHEMA = json.load(f)

with open(SCHEMAS / "glycan_v1.json") as f:
    GLYCAN_SCHEMA = json.load(f)

def risk_score(payload: dict) -> float:
    '''Very naive risk scoring stub.'''
    text = json.dumps(payload).lower()
    keys = ["weapon", "pathogen", "lethal", "lab protocol", "amplification"]
    score = 0.0
    for k in keys:
        if k in text:
            score += 0.25
    return min(score, 1.0)

def enforce_policy(payload: dict) -> dict:
    score = risk_score(payload)
    if score >= POLICY["enforcement"]["refusal_threshold"]:
        return {"status": "refused", "reason": "biosecurity risk", "ticket_id": f"POL-{uuid.uuid4()}", "risk": score}
    return {"status": "allowed", "risk": score}

def load_graph() -> Graph:
    g = Graph()
    if KG_SAMPLE.exists():
        g.parse(KG_SAMPLE, format="turtle")
    return g

async def initialize_platform():
    """Initialize platform components."""
    if HAS_PLATFORM_COMPONENTS:
        try:
            logger.info("Initializing platform components...")
            platform_components['data_coordinator'] = DataCoordinator()
            platform_components['reasoning_orchestrator'] = ReasoningOrchestrator(
                platform_components['data_coordinator']
            )
            logger.info("Platform components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize platform components: {e}")
    else:
        logger.info("Running in basic mode without platform components")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await initialize_platform()
    yield
    # Shutdown
    platform_components.clear()

app = FastAPI(
    title="Glycoinformatics AI Platform",
    description="A comprehensive multimodal AI system for glycan analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    up = {"status": "ok", "uptime_s": round(time.time()-APP_START, 2)}
    return up

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# SPARQL and Knowledge Graph Components
try:
    from glycokg.query.sparql_utils import SPARQLQueryManager, QueryResult
    from glycokg.ontology.glyco_ontology import GlycoOntology
    from glycokg.integration.glytoucan_client import GlyTouCanClient
    HAS_KG_COMPONENTS = True
except ImportError:
    HAS_KG_COMPONENTS = False
    logger.warning("Knowledge graph components not available")

class SPARQLQueryRequest(BaseModel):
    """SPARQL query request model."""
    query: str = Field(..., description="SPARQL query string or predefined query name")
    format: str = Field("json", description="Response format: json, table, csv, markdown")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    use_cache: bool = Field(True, description="Use cached results if available")
    timeout: int = Field(30, description="Query timeout in seconds")

class GlycanLookupRequest(BaseModel):
    """Glycan lookup request model."""
    identifier: str = Field(..., description="Glycan identifier (GlyTouCan ID, WURCS, etc.)")
    identifier_type: str = Field("glytoucan_id", description="Type of identifier")
    include_associations: bool = Field(True, description="Include protein associations")
    include_spectra: bool = Field(False, description="Include MS spectra data")

@app.post("/kg/sparql")
async def sparql_query_endpoint(request: SPARQLQueryRequest):
    """
    Execute SPARQL queries against the glycoinformatics knowledge graph.
    
    Supports both predefined queries and custom SPARQL with parameter substitution,
    result caching, and multiple output formats.
    """
    route = "/kg/sparql"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        if not HAS_KG_COMPONENTS:
            return JSONResponse({
                "success": False,
                "error": "Knowledge graph components not available",
                "message": "Install rdflib and SPARQLWrapper to enable SPARQL queries"
            }, status_code=503)
        
        try:
            # Initialize SPARQL query manager if not already done
            if 'sparql_manager' not in platform_components:
                # Load local knowledge graph
                local_graph = load_graph()
                platform_components['sparql_manager'] = SPARQLQueryManager(
                    local_graph=local_graph,
                    query_cache_size=100
                )
            
            sparql_manager = platform_components['sparql_manager']
            
            # Validate query syntax first
            validation = sparql_manager.validate_query(request.query)
            if not validation["valid"]:
                return JSONResponse({
                    "success": False,
                    "error": "Invalid SPARQL syntax",
                    "details": validation["error"],
                    "message": validation["message"]
                }, status_code=400)
            
            # Execute query
            result = sparql_manager.execute_query(
                query=request.query,
                use_cache=request.use_cache,
                timeout=request.timeout,
                **request.parameters
            )
            
            if result.error:
                return JSONResponse({
                    "success": False,
                    "error": "Query execution failed",
                    "details": result.error,
                    "execution_time": result.execution_time
                }, status_code=500)
            
            # Format results according to requested format
            formatted_result = sparql_manager.format_results(result, request.format)
            
            return {
                "success": True,
                "data": formatted_result["data"],
                "metadata": {
                    "result_count": result.result_count,
                    "execution_time": result.execution_time,
                    "format": request.format,
                    "cached": request.use_cache and len(sparql_manager.query_cache) > 0,
                    "timestamp": result.timestamp.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return JSONResponse({
                "success": False,
                "error": f"SPARQL query execution failed: {str(e)}"
            }, status_code=500)

@app.get("/kg/queries")
async def list_predefined_queries():
    """List available predefined SPARQL queries."""
    route = "/kg/queries"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        if not HAS_KG_COMPONENTS:
            return JSONResponse({
                "success": False,
                "error": "Knowledge graph components not available"
            }, status_code=503)
        
        try:
            if 'sparql_manager' not in platform_components:
                # Initialize with empty graph for query listing
                platform_components['sparql_manager'] = SPARQLQueryManager()
            
            sparql_manager = platform_components['sparql_manager']
            queries = sparql_manager.get_predefined_queries()
            
            query_info = []
            for query_name in queries:
                template = sparql_manager.get_query_template(query_name)
                
                # Extract description from query comments
                lines = template.split('\n') if template else []
                description = ""
                for line in lines[:5]:  # Check first 5 lines for description
                    if line.strip().startswith('#') and not line.strip().startswith('# '):
                        description = line.strip()[1:].strip()
                        break
                
                query_info.append({
                    "name": query_name,
                    "description": description or f"Predefined query: {query_name}",
                    "has_parameters": "${" in template if template else False
                })
            
            return {
                "success": True,
                "data": {
                    "queries": query_info,
                    "total_count": len(queries)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list queries: {e}")
            return JSONResponse({
                "success": False,
                "error": f"Failed to list predefined queries: {str(e)}"
            }, status_code=500)

@app.get("/kg/queries/{query_name}")
async def get_query_template(query_name: str):
    """Get template for a specific predefined query."""
    route = "/kg/queries/{query_name}"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        if not HAS_KG_COMPONENTS:
            return JSONResponse({
                "success": False,
                "error": "Knowledge graph components not available"
            }, status_code=503)
        
        try:
            if 'sparql_manager' not in platform_components:
                platform_components['sparql_manager'] = SPARQLQueryManager()
            
            sparql_manager = platform_components['sparql_manager']
            template = sparql_manager.get_query_template(query_name)
            
            if not template:
                return JSONResponse({
                    "success": False,
                    "error": f"Query '{query_name}' not found"
                }, status_code=404)
            
            # Parse parameters from template
            import re
            parameters = re.findall(r'\$\{(\w+)\}', template)
            
            return {
                "success": True,
                "data": {
                    "name": query_name,
                    "template": template,
                    "parameters": list(set(parameters))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get query template: {e}")
            return JSONResponse({
                "success": False,
                "error": f"Failed to get query template: {str(e)}"
            }, status_code=500)

@app.post("/kg/glycan/lookup")
async def lookup_glycan(request: GlycanLookupRequest):
    """
    Look up comprehensive glycan information from knowledge graph.
    
    Supports lookup by GlyTouCan ID, WURCS sequence, and other identifiers
    with optional inclusion of protein associations and experimental data.
    """
    route = "/kg/glycan/lookup"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        if not HAS_KG_COMPONENTS:
            return JSONResponse({
                "success": False,
                "error": "Knowledge graph components not available"
            }, status_code=503)
        
        try:
            # Initialize GlyTouCan client if needed
            if 'glytoucan_client' not in platform_components:
                platform_components['glytoucan_client'] = GlyTouCanClient()
            
            glytoucan_client = platform_components['glytoucan_client']
            
            # Look up glycan structure
            if request.identifier_type == "glytoucan_id":
                structure = glytoucan_client.get_structure_details(request.identifier)
            else:
                # For other identifier types, implement search logic
                return JSONResponse({
                    "success": False,
                    "error": f"Lookup by {request.identifier_type} not yet implemented"
                }, status_code=501)
            
            if not structure:
                return JSONResponse({
                    "success": False,
                    "error": f"Glycan '{request.identifier}' not found"
                }, status_code=404)
            
            result = {
                "glycan_info": {
                    "glytoucan_id": structure.glytoucan_id,
                    "wurcs_sequence": structure.wurcs_sequence,
                    "glycoct": structure.glycoct,
                    "iupac_extended": structure.iupac_extended,
                    "mass_monoisotopic": structure.mass_mono,
                    "mass_average": structure.mass_avg,
                    "composition": structure.composition
                }
            }
            
            # Add associations if requested
            if request.include_associations and 'sparql_manager' in platform_components:
                sparql_manager = platform_components['sparql_manager']
                
                # Query for protein associations
                assoc_query = f"""
                PREFIX glyco: <https://glycokg.org/ontology#>
                SELECT ?protein_id ?confidence ?tissue WHERE {{
                    ?glycan glyco:hasGlyTouCanID "{structure.glytoucan_id}" .
                    ?association glyco:hasGlycan ?glycan ;
                                glyco:hasProtein ?protein ;
                                glyco:hasConfidenceScore ?confidence .
                    ?protein glyco:hasUniProtID ?protein_id .
                    OPTIONAL {{
                        ?association glyco:expressedIn ?tissue_uri .
                        ?tissue_uri rdfs:label ?tissue .
                    }}
                }}
                ORDER BY DESC(?confidence)
                LIMIT 20
                """
                
                assoc_result = sparql_manager.execute_query(assoc_query)
                result["protein_associations"] = assoc_result.results
            
            # Add spectra data if requested
            if request.include_spectra and 'sparql_manager' in platform_components:
                sparql_manager = platform_components['sparql_manager']
                
                spectra_query = f"""
                PREFIX glyco: <https://glycokg.org/ontology#>
                SELECT ?spectrum_id ?precursor_mz ?charge ?collision_energy WHERE {{
                    ?glycan glyco:hasGlyTouCanID "{structure.glytoucan_id}" .
                    ?spectrum glyco:identifiesGlycan ?glycan ;
                             glyco:hasSpectrumID ?spectrum_id ;
                             glyco:hasPrecursorMZ ?precursor_mz ;
                             glyco:hasChargeState ?charge .
                    OPTIONAL {{ ?spectrum glyco:hasCollisionEnergy ?collision_energy }}
                }}
                LIMIT 10
                """
                
                spectra_result = sparql_manager.execute_query(spectra_query)
                result["ms_spectra"] = spectra_result.results
            
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Glycan lookup failed: {e}")
            return JSONResponse({
                "success": False,
                "error": f"Glycan lookup failed: {str(e)}"
            }, status_code=500)

@app.post("/kg/query")
def kg_query_legacy(request: Request, body: str = None):
    """Legacy SPARQL query endpoint for backward compatibility."""
    route = "/kg/query"
    REQUESTS.labels(route=route).inc()
    with LATENCY.labels(route=route).time():
        g = load_graph()
        query = body or ""
        try:
            res = g.query(query)
            rows = []
            for row in res:
                rows.append([str(x) for x in row])
            return {"rows": rows, "count": len(rows)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/kg/statistics")
async def get_kg_statistics():
    """Get knowledge graph statistics and metrics."""
    route = "/kg/statistics"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        try:
            stats = {
                "knowledge_graph": {
                    "local_graph_loaded": True,
                    "components_available": HAS_KG_COMPONENTS
                }
            }
            
            # Get local graph statistics
            local_graph = load_graph()
            stats["knowledge_graph"]["total_triples"] = len(local_graph)
            
            # Get ontology statistics if available
            if HAS_KG_COMPONENTS:
                try:
                    ontology = GlycoOntology()
                    ontology_stats = ontology.get_statistics()
                    stats["ontology"] = ontology_stats
                except Exception as e:
                    stats["ontology"] = {"error": str(e)}
                
                # Get GlyTouCan statistics if available
                try:
                    if 'glytoucan_client' not in platform_components:
                        platform_components['glytoucan_client'] = GlyTouCanClient()
                    
                    glytoucan_client = platform_components['glytoucan_client']
                    glytoucan_stats = glytoucan_client.get_statistics()
                    stats["glytoucan_repository"] = glytoucan_stats
                except Exception as e:
                    stats["glytoucan_repository"] = {"error": str(e)}
                
                # Get SPARQL manager statistics
                if 'sparql_manager' in platform_components:
                    sparql_manager = platform_components['sparql_manager']
                    sparql_stats = sparql_manager.get_statistics()
                    stats["sparql_manager"] = sparql_stats
            
            return {
                "success": True,
                "data": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get KG statistics: {e}")
            return JSONResponse({
                "success": False,
                "error": f"Failed to get statistics: {str(e)}"
            }, status_code=500)

class InferPayload(BaseModel):
    task: str | None = None
    text: str | None = None
    spectra: dict | None = None
    glycan: dict | None = None

@app.post("/llm/infer")
def llm_infer(payload: InferPayload):
    route = "/llm/infer"
    REQUESTS.labels(route=route).inc()
    with LATENCY.labels(route=route).time():
        data = payload.model_dump()
        # Policy gate
        policy = enforce_policy(data)
        if policy["status"] == "refused":
            return JSONResponse({"status":"refused","reason":policy["reason"],"ticket_id":policy["ticket_id"],"risk":policy["risk"]}, status_code=403)

        # Validate schemas if present
        try:
            if data.get("spectra"):
                validate(instance=data["spectra"], schema=SPECTRA_SCHEMA)
            if data.get("glycan"):
                validate(instance=data["glycan"], schema=GLYCAN_SCHEMA)
        except ValidationError as ve:
            return JSONResponse({"status":"invalid","error":str(ve)}, status_code=400)

        # OOD/uncertainty stub
        confidence = 0.62 if data.get("spectra") else 0.4
        if confidence < 0.35:
            return {"status":"abstain","confidence":confidence,"message":"Low confidence; please supply more evidence.","grounding":[]}

        # Mocked result with grounding
        return {
            "status":"ok",
            "task": data.get("task","qna"),
            "confidence": confidence,
            "result": {"summary":"MVP result placeholder; integrate real models."},
            "grounding": ["glycan:GLYCAN_000001","protein:UniProtKB:P12345"]
        }

class GoTRequest(BaseModel):
    goal: str
    inputs: dict | None = None
    constraints: dict | None = None
    uncertainty: dict | None = None

@app.post("/got/plan")
def got_plan(payload: GoTRequest):
    route = "/got/plan"
    REQUESTS.labels(route=route).inc()
    with LATENCY.labels(route=route).time():
        data = payload.model_dump()
        policy = enforce_policy(data)
        if policy["status"] == "refused":
            return JSONResponse({"status":"refused","reason":policy["reason"],"ticket_id":policy["ticket_id"]}, status_code=403)

        steps = [
            {"id":"s1","op":"decompose_spectrum","inputs":["spectra"],"outputs":["candidate_fragments"],"confidence":0.86},
            {"id":"s2","op":"motif_lookup","inputs":["candidate_fragments"],"uses":"GlycoKG","outputs":["motif_hits"]},
            {"id":"s3","op":"pathway_consistency_check","inputs":["motif_hits","organism"],"outputs":["filtered_candidates"]},
            {"id":"s4","op":"llm_rationale","inputs":["filtered_candidates"],"outputs":["explanation"],"audit_log":True},
        ]
        return {"goal": data.get("goal"), "steps": steps, "metrics":{"beam_width":10,"max_steps":8}, "uncertainty":{"type":"mc_dropout","samples":20}}

# Enhanced API endpoints for comprehensive platform

class GlycanAnalysisRequest(BaseModel):
    """Enhanced glycan analysis request."""
    structure: str = Field(..., description="Glycan structure (WURCS, GlycoCT, or IUPAC)")
    structure_format: str = Field("WURCS", description="Structure format")
    analysis_types: List[str] = Field(default=["structure_analysis"], description="Analysis types")
    context: Dict[str, Any] = Field(default_factory=dict, description="Analysis context")

class ReasoningQueryRequest(BaseModel):
    """Reasoning query request."""
    query: str = Field(..., description="Natural language query")
    structure: Optional[str] = Field(None, description="Optional glycan structure")
    reasoning_tasks: List[str] = Field(default=["structure_analysis"], description="Reasoning tasks")
    context: Dict[str, Any] = Field(default_factory=dict, description="Query context")

@app.post("/structure/analyze")
async def analyze_structure_comprehensive(request: GlycanAnalysisRequest):
    """Comprehensive glycan structure analysis using integrated platform."""
    route = "/structure/analyze"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        start_time = time.time()
        
        if HAS_PLATFORM_COMPONENTS and 'reasoning_orchestrator' in platform_components:
            try:
                # Use full platform capabilities
                orchestrator = platform_components['reasoning_orchestrator']
                
                reasoning_request = ReasoningRequest(
                    request_id=f"structure_analysis_{int(time.time())}",
                    structure=request.structure,
                    reasoning_tasks=[ReasoningType(task) for task in request.analysis_types],
                    context=request.context
                )
                
                result = await orchestrator.process_reasoning_request(reasoning_request)
                
                return {
                    "success": True,
                    "data": {
                        "analysis_result": result.dict() if hasattr(result, 'dict') else str(result),
                        "structure_format": request.structure_format,
                        "analysis_types": request.analysis_types
                    },
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                logger.error(f"Comprehensive analysis failed: {e}")
                return JSONResponse(
                    {"success": False, "error": f"Analysis failed: {str(e)}"}, 
                    status_code=500
                )
        else:
            # Fallback to basic analysis
            return {
                "success": True,
                "data": {
                    "analysis_result": "Basic analysis placeholder - platform components not loaded",
                    "structure": request.structure,
                    "structure_format": request.structure_format,
                    "note": "Running in basic mode"
                },
                "execution_time": time.time() - start_time
            }

@app.post("/reasoning/query")
async def process_reasoning_query(request: ReasoningQueryRequest):
    """Process natural language reasoning queries about glycans."""
    route = "/reasoning/query"
    REQUESTS.labels(route=route).inc()
    
    with LATENCY.labels(route=route).time():
        start_time = time.time()
        
        if HAS_PLATFORM_COMPONENTS and 'reasoning_orchestrator' in platform_components:
            try:
                orchestrator = platform_components['reasoning_orchestrator']
                
                reasoning_request = ReasoningRequest(
                    request_id=f"query_{int(time.time())}",
                    text_description=request.query,
                    structure=request.structure,
                    reasoning_tasks=[ReasoningType(task) for task in request.reasoning_tasks],
                    context=request.context
                )
                
                result = await orchestrator.process_reasoning_request(reasoning_request)
                
                return {
                    "success": True,
                    "data": {
                        "reasoning_result": result.dict() if hasattr(result, 'dict') else str(result),
                        "query": request.query,
                        "reasoning_tasks": request.reasoning_tasks
                    },
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                logger.error(f"Reasoning query failed: {e}")
                return JSONResponse(
                    {"success": False, "error": f"Query failed: {str(e)}"}, 
                    status_code=500
                )
        else:
            # Basic fallback
            return {
                "success": True,
                "data": {
                    "reasoning_result": f"Basic response to: {request.query}",
                    "note": "Running in basic mode - full reasoning not available"
                },
                "execution_time": time.time() - start_time
            }

@app.get("/platform/status")
async def get_platform_status():
    """Get comprehensive platform status."""
    status = {
        "platform_mode": "comprehensive" if HAS_PLATFORM_COMPONENTS else "basic",
        "components_loaded": list(platform_components.keys()),
        "uptime": time.time() - APP_START,
        "version": "0.1.0",
        "services": {}
    }
    
    # Check component health
    for component_name, component in platform_components.items():
        try:
            status["services"][component_name] = "healthy" if component else "unhealthy"
        except Exception as e:
            status["services"][component_name] = f"error: {str(e)}"
    
    return status