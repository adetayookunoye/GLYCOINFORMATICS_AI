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

@app.post("/kg/query")
def kg_query(request: Request, body: str = None):
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

# Import advanced models and inference engine
try:
    from .models import (
        Spec2StructRequest, Spec2StructResponse,
        Structure2SpecRequest, Structure2SpecResponse,
        ExplainRequest, ExplainResponse,
        RetrievalRequest, RetrievalResponse,
        APIError, TaskType, SpectraType, StructureFormat
    )
    from .inference import AdvancedGlycoLLMInference
    HAS_ADVANCED_API = True
except ImportError as e:
    logger.warning(f"Advanced API components not available: {e}")
    HAS_ADVANCED_API = False

# Global inference engine
inference_engine = None

async def get_inference_engine():
    """Get or initialize the inference engine."""
    global inference_engine
    if inference_engine is None:
        try:
            inference_engine = AdvancedGlycoLLMInference(device="auto")
            logger.info("AdvancedGlycoLLMInference engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise HTTPException(status_code=500, detail="Inference engine initialization failed")
    return inference_engine

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
        "advanced_api": "available" if HAS_ADVANCED_API else "unavailable",
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

# ========================================================================================
# SOPHISTICATED GLYCOLLM API ENDPOINTS
# ========================================================================================

if HAS_ADVANCED_API:
    
    @app.post("/glycollm/spec2struct", response_model=Spec2StructResponse)
    async def spec2struct_endpoint(request: Spec2StructRequest):
        """
        Advanced spectrum-to-structure prediction endpoint.
        
        Predicts glycan structures from mass spectra with uncertainty quantification,
        knowledge graph grounding, and comprehensive rationale generation.
        """
        route = "/glycollm/spec2struct"
        REQUESTS.labels(route=route).inc()
        
        with LATENCY.labels(route=route).time():
            start_time = time.time()
            
            try:
                # Policy enforcement
                policy = enforce_policy(request.model_dump())
                if policy["status"] == "refused":
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error_code": "POLICY_VIOLATION",
                            "message": policy["reason"],
                            "ticket_id": policy["ticket_id"]
                        }
                    )
                
                # Get inference engine
                engine = await get_inference_engine()
                
                # Perform inference
                candidates, uncertainty, rationale, next_steps = await engine.spec2struct(
                    spectra_data=request.spectra,
                    spectra_type=request.spectra_type,
                    max_candidates=request.max_candidates,
                    confidence_threshold=request.confidence_threshold,
                    include_grounding=request.include_grounding
                )
                
                return Spec2StructResponse(
                    candidates=candidates,
                    uncertainty=uncertainty,
                    rationale=rationale,
                    next_steps=next_steps,
                    processing_time=time.time() - start_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Spec2struct endpoint error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "INFERENCE_ERROR",
                        "message": f"Inference failed: {str(e)}",
                        "suggestion": "Check input format and try again"
                    }
                )
    
    @app.post("/glycollm/structure2spec", response_model=Structure2SpecResponse)
    async def structure2spec_endpoint(request: Structure2SpecRequest):
        """
        Advanced structure-to-spectrum prediction endpoint.
        
        Predicts mass spectra from glycan structures with fragmentation modeling,
        uncertainty estimation, and experimental validation recommendations.
        """
        route = "/glycollm/structure2spec"
        REQUESTS.labels(route=route).inc()
        
        with LATENCY.labels(route=route).time():
            start_time = time.time()
            
            try:
                # Policy enforcement
                policy = enforce_policy(request.model_dump())
                if policy["status"] == "refused":
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error_code": "POLICY_VIOLATION", 
                            "message": policy["reason"],
                            "ticket_id": policy["ticket_id"]
                        }
                    )
                
                # Get inference engine
                engine = await get_inference_engine()
                
                # Perform inference
                candidates, uncertainty, rationale, next_steps = await engine.structure2spec(
                    structure=request.structure,
                    structure_format=request.structure_format,
                    fragmentation_type=request.fragmentation_type,
                    charge_states=request.charge_states
                )
                
                return Structure2SpecResponse(
                    candidates=candidates,
                    uncertainty=uncertainty,
                    rationale=rationale,
                    next_steps=next_steps,
                    processing_time=time.time() - start_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Structure2spec endpoint error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "INFERENCE_ERROR",
                        "message": f"Inference failed: {str(e)}",
                        "suggestion": "Verify structure format and parameters"
                    }
                )
    
    @app.post("/glycollm/explain", response_model=ExplainResponse)
    async def explain_endpoint(request: ExplainRequest):
        """
        Advanced glycan explanation endpoint.
        
        Generates comprehensive explanations for glycan structures, spectra, or
        specific questions with knowledge graph integration and literature support.
        """
        route = "/glycollm/explain"
        REQUESTS.labels(route=route).inc()
        
        with LATENCY.labels(route=route).time():
            start_time = time.time()
            
            try:
                # Policy enforcement
                policy = enforce_policy(request.model_dump())
                if policy["status"] == "refused":
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error_code": "POLICY_VIOLATION",
                            "message": policy["reason"], 
                            "ticket_id": policy["ticket_id"]
                        }
                    )
                
                # Get inference engine
                engine = await get_inference_engine()
                
                # Perform inference
                explanation, key_insights, uncertainty, grounding, rationale, next_steps = await engine.explain(
                    structure=request.structure,
                    spectra=request.spectra,
                    question=request.question,
                    context=request.context,
                    explanation_depth=request.explanation_depth
                )
                
                return ExplainResponse(
                    explanation=explanation,
                    key_insights=key_insights,
                    uncertainty=uncertainty,
                    grounding=grounding,
                    rationale=rationale,
                    next_steps=next_steps,
                    processing_time=time.time() - start_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Explain endpoint error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "INFERENCE_ERROR",
                        "message": f"Explanation generation failed: {str(e)}",
                        "suggestion": "Provide more specific input or context"
                    }
                )
    
    @app.post("/glycollm/retrieval", response_model=RetrievalResponse) 
    async def retrieval_endpoint(request: RetrievalRequest):
        """
        Advanced glycan retrieval endpoint.
        
        Performs intelligent database search with semantic similarity, multi-modal
        queries, and confidence-ranked results from multiple glycan databases.
        """
        route = "/glycollm/retrieval"
        REQUESTS.labels(route=route).inc()
        
        with LATENCY.labels(route=route).time():
            start_time = time.time()
            
            try:
                # Policy enforcement
                policy = enforce_policy(request.model_dump())
                if policy["status"] == "refused":
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error_code": "POLICY_VIOLATION",
                            "message": policy["reason"],
                            "ticket_id": policy["ticket_id"]
                        }
                    )
                
                # Get inference engine
                engine = await get_inference_engine()
                
                # Perform retrieval
                results, total_found, uncertainty, rationale, next_steps = await engine.retrieval(
                    query=request.query,
                    query_type=request.query_type,
                    max_results=request.max_results,
                    similarity_threshold=request.similarity_threshold,
                    databases=request.databases
                )
                
                # Convert results to RetrievalResult objects
                from .models import RetrievalResult
                formatted_results = [
                    RetrievalResult(**result) for result in results
                ]
                
                return RetrievalResponse(
                    query=request.query,
                    results=formatted_results,
                    total_found=total_found,
                    uncertainty=uncertainty,
                    rationale=rationale,
                    next_steps=next_steps,
                    processing_time=time.time() - start_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Retrieval endpoint error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "RETRIEVAL_ERROR", 
                        "message": f"Database retrieval failed: {str(e)}",
                        "suggestion": "Refine query or check database connectivity"
                    }
                )
    
    @app.get("/glycollm/capabilities")
    async def get_glycollm_capabilities():
        """Get comprehensive information about GlycoLLM capabilities and features."""
        return {
            "version": "0.1.0",
            "model_type": "multimodal_transformer",
            "supported_tasks": [
                {
                    "name": "spec2struct",
                    "description": "Mass spectrum to glycan structure prediction",
                    "input_formats": ["peaks", "mzML", "JSON"],
                    "output_formats": ["WURCS", "GlycoCT", "IUPAC"],
                    "features": ["uncertainty_quantification", "kg_grounding", "rationale"]
                },
                {
                    "name": "structure2spec",
                    "description": "Glycan structure to mass spectrum prediction", 
                    "input_formats": ["WURCS", "GlycoCT", "IUPAC"],
                    "fragmentation_types": ["MS/MS", "CID", "HCD", "ETD", "ECD"],
                    "features": ["multi_charge_states", "fragment_annotation", "intensity_prediction"]
                },
                {
                    "name": "explain",
                    "description": "Comprehensive glycan analysis and explanation",
                    "input_types": ["structure", "spectrum", "question", "mixed"],
                    "features": ["literature_integration", "pathway_analysis", "biological_context"]
                },
                {
                    "name": "retrieval", 
                    "description": "Intelligent glycan database search",
                    "databases": ["GlyTouCan", "GlyGen", "GlycoPOST", "local_kg"],
                    "query_types": ["structure", "text", "similarity", "mixed"],
                    "features": ["semantic_search", "similarity_ranking", "cross_database"]
                }
            ],
            "uncertainty_methods": [
                "conformal_prediction",
                "selective_prediction", 
                "confidence_calibration",
                "epistemic_aleatoric_decomposition"
            ],
            "knowledge_integration": [
                "glyco_kg_grounding",
                "biosynthetic_pathways", 
                "literature_references",
                "organism_context"
            ],
            "performance": {
                "max_concurrent_requests": 10,
                "typical_response_time": "2-5 seconds",
                "confidence_threshold_range": [0.0, 1.0],
                "supported_gpus": ["CUDA", "MPS", "CPU_fallback"]
            }
        }

else:
    # Fallback endpoints when advanced API is not available
    @app.post("/glycollm/{task}")
    async def glycollm_fallback(task: str):
        return {
            "error": "Advanced GlycoLLM API not available",
            "message": "Please install required dependencies",
            "available_endpoint": "/llm/infer"
        }