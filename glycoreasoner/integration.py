"""
GlycoGOT Integration Coordinator

This module provides a unified interface for coordinating between the knowledge graph,
AI models, and reasoning engine to create seamless glycoinformatics workflows.
Supports multi-modal analysis, automated pipelines, and intelligent task orchestration.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import traceback

# Optional imports for enhanced functionality
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HAS_CONCURRENT = True
except ImportError:
    HAS_CONCURRENT = False

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of glycoinformatics tasks"""
    STRUCTURE_IDENTIFICATION = "structure_identification"
    FUNCTION_PREDICTION = "function_prediction"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    PATHWAY_ANALYSIS = "pathway_analysis"
    SIMILARITY_SEARCH = "similarity_search"
    LITERATURE_MINING = "literature_mining"
    MULTI_MODAL_ANALYSIS = "multi_modal_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"


class ComponentType(Enum):
    """Types of system components"""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    AI_MODEL = "ai_model"
    REASONING_ENGINE = "reasoning_engine"
    TOKENIZER = "tokenizer"
    DATASET = "dataset"
    API_CLIENT = "api_client"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskContext:
    """Context information for a glycoinformatics task"""
    task_id: str
    task_type: TaskType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = high, 5 = low
    timeout: float = 300.0  # 5 minutes default
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result of a glycoinformatics task"""
    task_id: str
    status: TaskStatus
    results: Dict[str, Any]
    execution_time: float
    component_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """A single step in a workflow"""
    step_id: str
    component_type: ComponentType
    operation: str
    inputs: Dict[str, Any]
    outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False


@dataclass
class Workflow:
    """A complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    task_type: TaskType
    expected_duration: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentInterface(ABC):
    """Abstract interface for system components"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component"""
        pass
        
    @abstractmethod
    def process(self, 
                operation: str,
                inputs: Dict[str, Any],
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data through the component"""
        pass
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Cleanup component resources"""
        pass


class KnowledgeGraphComponent(ComponentInterface):
    """Knowledge graph component wrapper"""
    
    def __init__(self):
        self.ontology = None
        self.query_manager = None
        self.glytoucan_client = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize knowledge graph components"""
        try:
            # Import components
            from ..glycokg.ontology.glyco_ontology import GlycoOntology
            from ..glycokg.query.sparql_utils import SPARQLQueryManager
            from ..glycokg.integration.glytoucan_client import GlyTouCanClient
            
            # Initialize ontology
            self.ontology = GlycoOntology()
            
            # Initialize query manager
            self.query_manager = SPARQLQueryManager(
                local_graph=self.ontology.graph,
                cache_enabled=config.get('enable_cache', True)
            )
            
            # Initialize GlyTouCan client
            self.glytoucan_client = GlyTouCanClient()
            
            self.is_initialized = True
            logger.info("Knowledge graph component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph component: {e}")
            return False
            
    def process(self, 
                operation: str,
                inputs: Dict[str, Any],
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process knowledge graph operations"""
        
        if not self.is_initialized:
            return {"error": "Component not initialized"}
            
        parameters = parameters or {}
        
        try:
            if operation == "sparql_query":
                query = inputs.get("query", "")
                result = self.query_manager.execute_query(query)
                return {
                    "results": result.results,
                    "result_count": result.result_count,
                    "execution_time": result.execution_time,
                    "error": result.error
                }
                
            elif operation == "glycan_lookup":
                glycan_id = inputs.get("glycan_id", "")
                if self.glytoucan_client:
                    glycan_info = self.glytoucan_client.get_glycan_info(glycan_id)
                    return {"glycan_info": glycan_info}
                else:
                    return {"error": "GlyTouCan client not available"}
                    
            elif operation == "add_glycan":
                glycan_data = inputs.get("glycan_data", {})
                glycan_id = glycan_data.get("id", "")
                wurcs = glycan_data.get("wurcs", "")
                mass = glycan_data.get("mass", 0.0)
                
                self.ontology.add_glycan(glycan_id, wurcs_sequence=wurcs, mass_mono=mass)
                return {"status": "added", "glycan_id": glycan_id}
                
            elif operation == "search_glycans":
                search_params = inputs.get("search_params", {})
                # Implement glycan search logic
                query = self._build_search_query(search_params)
                result = self.query_manager.execute_query(query)
                return {
                    "matching_glycans": result.results,
                    "count": result.result_count
                }
                
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Knowledge graph operation failed: {str(e)}"}
            
    def _build_search_query(self, search_params: Dict[str, Any]) -> str:
        """Build SPARQL query for glycan search"""
        
        base_query = f"""
        PREFIX glyco: <{self.ontology.GLYCO}>
        SELECT ?glycan ?id ?mass ?wurcs WHERE {{
            ?glycan a glyco:Glycan .
            ?glycan glyco:hasGlyTouCanID ?id .
        """
        
        conditions = []
        
        if "min_mass" in search_params:
            conditions.append(f"?glycan glyco:hasMonoisotopicMass ?mass . FILTER(?mass >= {search_params['min_mass']})")
            
        if "max_mass" in search_params:
            conditions.append(f"?glycan glyco:hasMonoisotopicMass ?mass . FILTER(?mass <= {search_params['max_mass']})")
            
        if "monosaccharide" in search_params:
            mono = search_params["monosaccharide"]
            conditions.append(f"?glycan glyco:hasWURCS ?wurcs . FILTER(CONTAINS(LCASE(?wurcs), LCASE('{mono}')))")
            
        if conditions:
            base_query += " " + " ".join(conditions)
            
        base_query += " }"
        
        return base_query
        
    def get_status(self) -> Dict[str, Any]:
        """Get knowledge graph component status"""
        status = {
            "initialized": self.is_initialized,
            "ontology_loaded": self.ontology is not None,
            "query_manager_ready": self.query_manager is not None,
            "glytoucan_client_ready": self.glytoucan_client is not None
        }
        
        if self.ontology:
            status["ontology_triples"] = len(self.ontology.graph)
            
        return status
        
    def cleanup(self):
        """Cleanup knowledge graph resources"""
        self.ontology = None
        self.query_manager = None
        self.glytoucan_client = None
        self.is_initialized = False


class AIModelComponent(ComponentInterface):
    """AI model component wrapper"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.dataset_builder = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize AI model components"""
        try:
            # Import AI components
            from ..glycollm.tokenization.glyco_tokenizer import GlycoTokenizer
            from ..glycollm.models.glycollm import GlycoLLM, GlycoLLMConfig
            from ..glycollm.data.multimodal_dataset import MultimodalDatasetBuilder
            
            # Initialize tokenizer
            self.tokenizer = GlycoTokenizer()
            
            # Initialize model with config
            model_config = GlycoLLMConfig(
                vocab_size=config.get('vocab_size', 50000),
                d_model=config.get('d_model', 768),
                n_layers=config.get('n_layers', 12),
                n_heads=config.get('n_heads', 12)
            )
            
            try:
                self.model = GlycoLLM(model_config)
                logger.info("GlycoLLM model initialized")
            except Exception as e:
                logger.warning(f"Could not initialize full model: {e}")
                self.model = None
                
            # Initialize dataset builder
            self.dataset_builder = MultimodalDatasetBuilder()
            
            self.is_initialized = True
            logger.info("AI model component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI model component: {e}")
            return False
            
    def process(self, 
                operation: str,
                inputs: Dict[str, Any],
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process AI model operations"""
        
        if not self.is_initialized:
            return {"error": "Component not initialized"}
            
        parameters = parameters or {}
        
        try:
            if operation == "tokenize":
                text = inputs.get("text", "")
                wurcs = inputs.get("wurcs", "")
                spectra = inputs.get("spectra", [])
                
                result = {}
                
                if text and hasattr(self.tokenizer, 'encode_text'):
                    result["text_tokens"] = self.tokenizer.encode_text(text)
                    
                if wurcs and hasattr(self.tokenizer, 'encode_wurcs'):
                    result["wurcs_tokens"] = self.tokenizer.encode_wurcs(wurcs)
                    
                if spectra and hasattr(self.tokenizer, 'encode_spectra'):
                    result["spectra_tokens"] = self.tokenizer.encode_spectra(spectra)
                    
                return result
                
            elif operation == "predict_structure":
                input_data = inputs.get("input_data", {})
                
                if self.model is None:
                    return {"error": "Model not available"}
                    
                # Simplified structure prediction
                return {
                    "predicted_structure": "WURCS=predicted_structure",
                    "confidence": 0.85,
                    "note": "This is a mock prediction - full model inference would be implemented here"
                }
                
            elif operation == "generate_text":
                prompt = inputs.get("prompt", "")
                max_length = parameters.get("max_length", 100)
                
                if self.model is None:
                    return {"error": "Model not available"}
                    
                # Mock text generation
                return {
                    "generated_text": f"Generated response for: {prompt[:50]}...",
                    "note": "This is a mock generation - full model would be used here"
                }
                
            elif operation == "build_dataset":
                data_sources = inputs.get("data_sources", {})
                dataset_config = parameters.get("dataset_config", {})
                
                # Use dataset builder to process data
                dataset_info = self.dataset_builder.build_dataset(
                    text_sources=data_sources.get("text_sources", []),
                    spectra_sources=data_sources.get("spectra_sources", []),
                    structure_sources=data_sources.get("structure_sources", [])
                )
                
                return {
                    "dataset_info": dataset_info,
                    "status": "built"
                }
                
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": f"AI model operation failed: {str(e)}"}
            
    def get_status(self) -> Dict[str, Any]:
        """Get AI model component status"""
        return {
            "initialized": self.is_initialized,
            "tokenizer_ready": self.tokenizer is not None,
            "model_ready": self.model is not None,
            "dataset_builder_ready": self.dataset_builder is not None
        }
        
    def cleanup(self):
        """Cleanup AI model resources"""
        self.tokenizer = None
        self.model = None
        self.dataset_builder = None
        self.is_initialized = False


class ReasoningEngineComponent(ComponentInterface):
    """Reasoning engine component wrapper"""
    
    def __init__(self):
        self.reasoner = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize reasoning engine"""
        try:
            from .reasoning import GlycoGOTReasoner
            
            self.reasoner = GlycoGOTReasoner()
            self.is_initialized = True
            logger.info("Reasoning engine component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine component: {e}")
            return False
            
    def process(self, 
                operation: str,
                inputs: Dict[str, Any],
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process reasoning operations"""
        
        if not self.is_initialized:
            return {"error": "Component not initialized"}
            
        parameters = parameters or {}
        
        try:
            if operation == "reason":
                goal = inputs.get("goal", "")
                context = inputs.get("context", {})
                
                result = self.reasoner.reason(goal, context)
                
                return {
                    "reasoning_result": {
                        "goal": result.goal,
                        "conclusion": result.conclusion,
                        "confidence": result.confidence,
                        "steps": len(result.reasoning_chain),
                        "execution_time": result.execution_time,
                        "success": result.success
                    },
                    "explanation": self.reasoner.get_reasoning_explanation(result),
                    "alternative_hypotheses": result.alternative_hypotheses
                }
                
            elif operation == "generate_hypotheses":
                context = inputs.get("context", {})
                max_hypotheses = parameters.get("max_hypotheses", 5)
                
                hypotheses = self.reasoner.hypothesis_generator.generate_hypotheses(
                    context, max_hypotheses
                )
                
                return {"hypotheses": hypotheses}
                
            elif operation == "analyze_causality":
                observations = inputs.get("observations", [])
                target_effect = inputs.get("target_effect", "")
                
                causal_analysis = self.reasoner.causal_reasoner.analyze_causality(
                    observations, target_effect
                )
                
                return {"causal_analysis": causal_analysis}
                
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Reasoning operation failed: {str(e)}"}
            
    def get_status(self) -> Dict[str, Any]:
        """Get reasoning engine status"""
        status = {
            "initialized": self.is_initialized,
            "reasoner_ready": self.reasoner is not None
        }
        
        if self.reasoner:
            status["reasoning_history_count"] = len(self.reasoner.reasoning_history)
            
        return status
        
    def cleanup(self):
        """Cleanup reasoning engine resources"""
        self.reasoner = None
        self.is_initialized = False


class GlycoGOTIntegrator:
    """Main integration coordinator for GlycoGOT system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.components = {
            ComponentType.KNOWLEDGE_GRAPH: KnowledgeGraphComponent(),
            ComponentType.AI_MODEL: AIModelComponent(),
            ComponentType.REASONING_ENGINE: ReasoningEngineComponent()
        }
        
        # Task management
        self.active_tasks = {}
        self.completed_tasks = {}
        self.workflows = {}
        
        # Execution pool for concurrent tasks
        self.executor = ThreadPoolExecutor(max_workers=4) if HAS_CONCURRENT else None
        
        # Initialize predefined workflows
        self._initialize_workflows()
        
        logger.info("GlycoGOT Integrator initialized")
        
    def initialize_components(self) -> Dict[ComponentType, bool]:
        """Initialize all system components"""
        initialization_results = {}
        
        for component_type, component in self.components.items():
            component_config = self.config.get(component_type.value, {})
            success = component.initialize(component_config)
            initialization_results[component_type] = success
            
            if success:
                logger.info(f"{component_type.value} initialized successfully")
            else:
                logger.error(f"Failed to initialize {component_type.value}")
                
        return initialization_results
        
    def _initialize_workflows(self):
        """Initialize predefined workflows"""
        
        # Structure identification workflow
        self.workflows["structure_identification"] = Workflow(
            workflow_id="struct_id_001",
            name="Glycan Structure Identification",
            description="Identify glycan structure from mass spectral data",
            task_type=TaskType.STRUCTURE_IDENTIFICATION,
            steps=[
                WorkflowStep(
                    step_id="preprocess_spectra",
                    component_type=ComponentType.AI_MODEL,
                    operation="tokenize",
                    inputs={"spectra": "input_spectra"},
                    outputs=["preprocessed_spectra"]
                ),
                WorkflowStep(
                    step_id="search_database",
                    component_type=ComponentType.KNOWLEDGE_GRAPH,
                    operation="search_glycans",
                    inputs={"search_params": "mass_range"},
                    outputs=["candidate_structures"],
                    dependencies=["preprocess_spectra"]
                ),
                WorkflowStep(
                    step_id="predict_structure",
                    component_type=ComponentType.AI_MODEL,
                    operation="predict_structure",
                    inputs={"input_data": "preprocessed_spectra"},
                    outputs=["predicted_structure"],
                    dependencies=["preprocess_spectra"]
                ),
                WorkflowStep(
                    step_id="reason_about_structure",
                    component_type=ComponentType.REASONING_ENGINE,
                    operation="reason",
                    inputs={"goal": "identify_most_likely_structure", "context": "all_data"},
                    outputs=["reasoning_result"],
                    dependencies=["search_database", "predict_structure"]
                )
            ]
        )
        
        # Function prediction workflow
        self.workflows["function_prediction"] = Workflow(
            workflow_id="func_pred_001",
            name="Glycan Function Prediction",
            description="Predict biological function of glycan structure",
            task_type=TaskType.FUNCTION_PREDICTION,
            steps=[
                WorkflowStep(
                    step_id="analyze_structure",
                    component_type=ComponentType.KNOWLEDGE_GRAPH,
                    operation="glycan_lookup",
                    inputs={"glycan_id": "input_glycan_id"},
                    outputs=["structural_info"]
                ),
                WorkflowStep(
                    step_id="predict_function",
                    component_type=ComponentType.AI_MODEL,
                    operation="predict_structure",  # Would be predict_function in real implementation
                    inputs={"input_data": "structural_info"},
                    outputs=["predicted_functions"],
                    dependencies=["analyze_structure"]
                ),
                WorkflowStep(
                    step_id="reason_about_function",
                    component_type=ComponentType.REASONING_ENGINE,
                    operation="reason",
                    inputs={"goal": "predict_biological_function", "context": "structure_and_predictions"},
                    outputs=["function_reasoning"],
                    dependencies=["predict_function"]
                )
            ]
        )
        
        # Multi-modal analysis workflow
        self.workflows["multi_modal_analysis"] = Workflow(
            workflow_id="multimodal_001",
            name="Multi-modal Glycan Analysis",
            description="Comprehensive analysis using multiple data modalities",
            task_type=TaskType.MULTI_MODAL_ANALYSIS,
            steps=[
                WorkflowStep(
                    step_id="tokenize_inputs",
                    component_type=ComponentType.AI_MODEL,
                    operation="tokenize",
                    inputs={"text": "input_text", "wurcs": "input_wurcs", "spectra": "input_spectra"},
                    outputs=["tokenized_data"]
                ),
                WorkflowStep(
                    step_id="query_knowledge_graph",
                    component_type=ComponentType.KNOWLEDGE_GRAPH,
                    operation="sparql_query",
                    inputs={"query": "related_glycans_query"},
                    outputs=["kg_results"]
                ),
                WorkflowStep(
                    step_id="generate_hypotheses",
                    component_type=ComponentType.REASONING_ENGINE,
                    operation="generate_hypotheses",
                    inputs={"context": "multimodal_context"},
                    outputs=["hypotheses"],
                    dependencies=["tokenize_inputs", "query_knowledge_graph"]
                ),
                WorkflowStep(
                    step_id="comprehensive_reasoning",
                    component_type=ComponentType.REASONING_ENGINE,
                    operation="reason",
                    inputs={"goal": "comprehensive_multimodal_analysis", "context": "all_results"},
                    outputs=["final_analysis"],
                    dependencies=["generate_hypotheses"]
                )
            ]
        )
        
    def submit_task(self, task_context: TaskContext) -> str:
        """Submit a task for execution"""
        
        task_id = task_context.task_id
        self.active_tasks[task_id] = {
            "context": task_context,
            "status": TaskStatus.PENDING,
            "submitted_at": time.time(),
            "future": None
        }
        
        # Submit task for execution
        if self.executor:
            future = self.executor.submit(self._execute_task, task_context)
            self.active_tasks[task_id]["future"] = future
        else:
            # Execute synchronously if no executor
            result = self._execute_task(task_context)
            self._handle_task_completion(task_id, result)
            
        logger.info(f"Task {task_id} submitted for execution")
        return task_id
        
    def _execute_task(self, task_context: TaskContext) -> TaskResult:
        """Execute a single task"""
        
        task_id = task_context.task_id
        start_time = time.time()
        
        try:
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = TaskStatus.RUNNING
                
            # Determine execution strategy
            if task_context.task_type in [tt for tt in TaskType]:
                # Use workflow execution
                result = self._execute_workflow(task_context)
            else:
                # Direct component execution
                result = self._execute_direct(task_context)
                
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                results=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(f"Task {task_id} failed: {error_msg}")
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                results={},
                execution_time=execution_time,
                errors=[error_msg]
            )
            
    def _execute_workflow(self, task_context: TaskContext) -> Dict[str, Any]:
        """Execute a predefined workflow"""
        
        workflow_name = task_context.task_type.value
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
            
        workflow = self.workflows[workflow_name]
        workflow_results = {}
        step_outputs = {}
        
        logger.info(f"Executing workflow: {workflow.name}")
        
        # Execute workflow steps in dependency order
        executed_steps = set()
        
        while len(executed_steps) < len(workflow.steps):
            for step in workflow.steps:
                if step.step_id in executed_steps:
                    continue
                    
                # Check if dependencies are satisfied
                if all(dep in executed_steps for dep in step.dependencies):
                    # Execute step
                    step_result = self._execute_workflow_step(
                        step, task_context, step_outputs
                    )
                    
                    step_outputs[step.step_id] = step_result
                    workflow_results[step.step_id] = step_result
                    executed_steps.add(step.step_id)
                    
                    logger.info(f"Completed workflow step: {step.step_id}")
                    
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "step_results": workflow_results,
            "final_outputs": step_outputs
        }
        
    def _execute_workflow_step(self, 
                              step: WorkflowStep,
                              task_context: TaskContext,
                              previous_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        # Get component
        component = self.components.get(step.component_type)
        if not component:
            raise ValueError(f"Component not available: {step.component_type}")
            
        # Prepare inputs by resolving references
        resolved_inputs = {}
        for input_key, input_ref in step.inputs.items():
            if input_ref in task_context.input_data:
                resolved_inputs[input_key] = task_context.input_data[input_ref]
            elif input_ref in previous_outputs:
                resolved_inputs[input_key] = previous_outputs[input_ref]
            else:
                # Try to resolve from step outputs
                resolved_value = self._resolve_input_reference(input_ref, previous_outputs, task_context)
                if resolved_value is not None:
                    resolved_inputs[input_key] = resolved_value
                    
        # Execute component operation
        result = component.process(
            operation=step.operation,
            inputs=resolved_inputs,
            parameters=step.parameters
        )
        
        return result
        
    def _resolve_input_reference(self, 
                                reference: str,
                                previous_outputs: Dict[str, Any],
                                task_context: TaskContext) -> Any:
        """Resolve input references for workflow steps"""
        
        # Handle special references
        if reference == "all_data":
            return {
                "input_data": task_context.input_data,
                "previous_outputs": previous_outputs
            }
        elif reference == "multimodal_context":
            return {
                "inputs": task_context.input_data,
                "outputs": previous_outputs,
                "task_type": task_context.task_type.value
            }
        elif reference == "mass_range":
            # Extract mass range from input data
            if "mass" in task_context.input_data:
                mass = task_context.input_data["mass"]
                return {
                    "min_mass": mass - 5.0,  # ±5 Da range
                    "max_mass": mass + 5.0
                }
        elif reference == "related_glycans_query":
            # Build query for related glycans
            return "SELECT ?glycan ?id WHERE { ?glycan a glyco:Glycan . ?glycan glyco:hasGlyTouCanID ?id . }"
        elif reference == "identify_most_likely_structure":
            return "Identify the most likely glycan structure from available data"
        elif reference == "predict_biological_function":
            return "Predict the biological function of this glycan structure"
        elif reference == "comprehensive_multimodal_analysis":
            return "Perform comprehensive analysis using all available data modalities"
            
        return None
        
    def _execute_direct(self, task_context: TaskContext) -> Dict[str, Any]:
        """Execute task directly without workflow"""
        
        # Simple direct execution for custom tasks
        results = {}
        
        # Try to determine appropriate component based on input data
        if "query" in task_context.input_data:
            # Likely a knowledge graph query
            component = self.components[ComponentType.KNOWLEDGE_GRAPH]
            results["kg_result"] = component.process(
                "sparql_query",
                {"query": task_context.input_data["query"]}
            )
            
        if "text" in task_context.input_data or "wurcs" in task_context.input_data:
            # AI model processing
            component = self.components[ComponentType.AI_MODEL]
            results["ai_result"] = component.process(
                "tokenize",
                task_context.input_data
            )
            
        if "goal" in task_context.input_data:
            # Reasoning task
            component = self.components[ComponentType.REASONING_ENGINE]
            results["reasoning_result"] = component.process(
                "reason",
                task_context.input_data
            )
            
        return results
        
    def _handle_task_completion(self, task_id: str, result: TaskResult):
        """Handle task completion"""
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = result.status
            
        self.completed_tasks[task_id] = result
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            
        logger.info(f"Task {task_id} completed with status: {result.status}")
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task_info["status"].value,
                "submitted_at": task_info["submitted_at"],
                "context": asdict(task_info["context"])
            }
            
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": result.status.value,
                "results": result.results,
                "execution_time": result.execution_time,
                "errors": result.errors,
                "warnings": result.warnings
            }
            
        return None
        
    def list_active_tasks(self) -> List[str]:
        """List all active task IDs"""
        return list(self.active_tasks.keys())
        
    def list_completed_tasks(self) -> List[str]:
        """List all completed task IDs"""
        return list(self.completed_tasks.keys())
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        component_status = {}
        for component_type, component in self.components.items():
            component_status[component_type.value] = component.get_status()
            
        return {
            "components": component_status,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "available_workflows": list(self.workflows.keys()),
            "executor_available": self.executor is not None
        }
        
    def create_custom_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a custom workflow from definition"""
        
        workflow = Workflow(
            workflow_id=workflow_definition["workflow_id"],
            name=workflow_definition["name"],
            description=workflow_definition["description"],
            task_type=TaskType(workflow_definition["task_type"]),
            steps=[
                WorkflowStep(**step_def) for step_def in workflow_definition["steps"]
            ]
        )
        
        self.workflows[workflow.workflow_id] = workflow
        
        logger.info(f"Custom workflow created: {workflow.workflow_id}")
        return workflow.workflow_id
        
    def shutdown(self):
        """Shutdown the integrator and cleanup resources"""
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task(task_id)
            
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            
        # Cleanup components
        for component in self.components.values():
            component.cleanup()
            
        logger.info("GlycoGOT Integrator shutdown complete")
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task"""
        
        if task_id not in self.active_tasks:
            return False
            
        task_info = self.active_tasks[task_id]
        
        if task_info["future"] and self.executor:
            task_info["future"].cancel()
            
        task_info["status"] = TaskStatus.CANCELLED
        
        # Move to completed tasks
        self.completed_tasks[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            results={},
            execution_time=time.time() - task_info["submitted_at"]
        )
        
        del self.active_tasks[task_id]
        
        logger.info(f"Task {task_id} cancelled")
        return True


# Convenience functions
def create_structure_identification_task(mass_spectrum: Dict[str, Any],
                                       additional_data: Dict[str, Any] = None) -> TaskContext:
    """Create a structure identification task"""
    
    input_data = {
        "input_spectra": mass_spectrum,
        "mass": mass_spectrum.get("precursor_mass", 0.0)
    }
    
    if additional_data:
        input_data.update(additional_data)
        
    return TaskContext(
        task_id=f"struct_id_{int(time.time())}",
        task_type=TaskType.STRUCTURE_IDENTIFICATION,
        input_data=input_data
    )


def create_function_prediction_task(glycan_id: str,
                                  additional_context: Dict[str, Any] = None) -> TaskContext:
    """Create a function prediction task"""
    
    input_data = {
        "input_glycan_id": glycan_id
    }
    
    if additional_context:
        input_data.update(additional_context)
        
    return TaskContext(
        task_id=f"func_pred_{int(time.time())}",
        task_type=TaskType.FUNCTION_PREDICTION,
        input_data=input_data
    )


def create_multimodal_analysis_task(text: str = "",
                                  wurcs: str = "",
                                  spectra: List[float] = None) -> TaskContext:
    """Create a multi-modal analysis task"""
    
    input_data = {}
    
    if text:
        input_data["input_text"] = text
    if wurcs:
        input_data["input_wurcs"] = wurcs
    if spectra:
        input_data["input_spectra"] = spectra
        
    return TaskContext(
        task_id=f"multimodal_{int(time.time())}",
        task_type=TaskType.MULTI_MODAL_ANALYSIS,
        input_data=input_data
    )


# Example usage and testing
def demo_integration():
    """Demonstrate GlycoGOT integration capabilities"""
    
    print("=== GlycoGOT Integration Demo ===\n")
    
    # Initialize integrator
    integrator = GlycoGOTIntegrator()
    
    # Initialize components
    init_results = integrator.initialize_components()
    print("Component Initialization:")
    for component_type, success in init_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {component_type.value}")
    print()
    
    # Get system status
    system_status = integrator.get_system_status()
    print("System Status:")
    print(json.dumps(system_status, indent=2, default=str))
    print()
    
    # Create and submit tasks
    tasks = []
    
    # Structure identification task
    struct_task = create_structure_identification_task({
        "precursor_mass": 365.1,
        "fragments": [203.1, 365.1, 542.2]
    })
    task_id_1 = integrator.submit_task(struct_task)
    tasks.append(task_id_1)
    
    # Function prediction task
    func_task = create_function_prediction_task(
        "G12345",
        {"protein_target": "CD22"}
    )
    task_id_2 = integrator.submit_task(func_task)
    tasks.append(task_id_2)
    
    # Multi-modal analysis task
    multimodal_task = create_multimodal_analysis_task(
        text="N-linked glycan with sialic acid",
        wurcs="WURCS=2.0,3,2,/1=1-1x2h3h4h6h/2=6d2h5n",
        spectra=[203.1, 365.1, 542.2]
    )
    task_id_3 = integrator.submit_task(multimodal_task)
    tasks.append(task_id_3)
    
    print(f"Submitted {len(tasks)} tasks for execution")
    
    # Wait for completion (in real usage would use async or polling)
    time.sleep(2)
    
    # Check task results
    print("\nTask Results:")
    for task_id in tasks:
        status = integrator.get_task_status(task_id)
        if status:
            print(f"\nTask {task_id}:")
            print(f"  Status: {status['status']}")
            if 'results' in status:
                print(f"  Results: {len(status['results'])} components")
            if 'execution_time' in status:
                print(f"  Execution time: {status['execution_time']:.2f}s")
                
    # Cleanup
    integrator.shutdown()
    
    return integrator


if __name__ == "__main__":
    # Run demonstration
    demo_integrator = demo_integration()