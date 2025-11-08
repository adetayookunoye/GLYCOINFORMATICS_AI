""""""

Integration Module for Graph of Thoughts with Glycoinformatics PlatformGlycoGOT Integration Coordinator

=======================================================================

This module provides a unified interface for coordinating between the knowledge graph,

This module provides integration between the Graph of Thoughts reasoning engineAI models, and reasoning engine to create seamless glycoinformatics workflows.

and the broader glycoinformatics AI platform, including LLM integration,Supports multi-modal analysis, automated pipelines, and intelligent task orchestration.

data pipeline connections, and API endpoints."""

"""

import asyncio

import asyncioimport logging

import loggingimport json

import jsonimport time

import timefrom typing import Dict, List, Optional, Tuple, Union, Any, Callable

from typing import Dict, List, Optional, Any, Callable, Unionfrom dataclasses import dataclass, field, asdict

from dataclasses import dataclassfrom enum import Enum

from enum import Enumfrom abc import ABC, abstractmethod

import uuidimport traceback



# Core GoT imports# Optional imports for enhanced functionality

from .graph_reasoning import (try:

    GraphOfThoughts, ThoughtNode, ThoughtGraph, ThoughtType, ThoughtStatus    from concurrent.futures import ThreadPoolExecutor, as_completed

)    HAS_CONCURRENT = True

from .thought_generators import (except ImportError:

    create_standard_generators, generators_to_functions,     HAS_CONCURRENT = False

    DeductiveReasoningGenerator, HypothesisGenerator

)logger = logging.getLogger(__name__)

from .search_algorithms import (

    SearchAlgorithmFactory, SearchStrategy, SearchResult,

    create_goal_condition_from_content, create_goal_condition_from_typeclass TaskType(Enum):

)    """Types of glycoinformatics tasks"""

from .evaluation import (    STRUCTURE_IDENTIFICATION = "structure_identification"

    PathEvaluator, GraphEvaluator, ReasoningComparator,    FUNCTION_PREDICTION = "function_prediction"

    create_evaluation_report    BIOMARKER_DISCOVERY = "biomarker_discovery"

)    PATHWAY_ANALYSIS = "pathway_analysis"

    SIMILARITY_SEARCH = "similarity_search"

logger = logging.getLogger(__name__)    LITERATURE_MINING = "literature_mining"

    MULTI_MODAL_ANALYSIS = "multi_modal_analysis"

    HYPOTHESIS_GENERATION = "hypothesis_generation"

class IntegrationMode(Enum):

    """Modes of integration with the platform"""

    STANDALONE = "standalone"              # Independent reasoningclass ComponentType(Enum):

    LLM_ASSISTED = "llm_assisted"         # With LLM support    """Types of system components"""

    KNOWLEDGE_ENHANCED = "knowledge_enhanced"  # With knowledge graph    KNOWLEDGE_GRAPH = "knowledge_graph"

    FULL_PIPELINE = "full_pipeline"       # Complete integration    AI_MODEL = "ai_model"

    REASONING_ENGINE = "reasoning_engine"

    TOKENIZER = "tokenizer"

class ReasoningTask(Enum):    DATASET = "dataset"

    """Types of reasoning tasks in glycoinformatics"""    API_CLIENT = "api_client"

    STRUCTURE_ELUCIDATION = "structure_elucidation"

    PATHWAY_ANALYSIS = "pathway_analysis"

    FUNCTION_PREDICTION = "function_prediction"class TaskStatus(Enum):

    BIOMARKER_DISCOVERY = "biomarker_discovery"    """Task execution status"""

    THERAPEUTIC_DESIGN = "therapeutic_design"    PENDING = "pending"

    QUALITY_CONTROL = "quality_control"    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"

@dataclass    CANCELLED = "cancelled"

class ReasoningRequest:

    """Request for Graph of Thoughts reasoning"""

    task_id: str@dataclass

    task_type: ReasoningTaskclass TaskContext:

    input_data: Dict[str, Any]    """Context information for a glycoinformatics task"""

    goal: str    task_id: str

    context: Dict[str, Any]    task_type: TaskType

    constraints: Dict[str, Any] = None    input_data: Dict[str, Any]

    mode: IntegrationMode = IntegrationMode.STANDALONE    parameters: Dict[str, Any] = field(default_factory=dict)

        metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:    priority: int = 1  # 1 = high, 5 = low

        return {    timeout: float = 300.0  # 5 minutes default

            'task_id': self.task_id,    created_at: float = field(default_factory=time.time)

            'task_type': self.task_type.value,

            'input_data': self.input_data,

            'goal': self.goal,@dataclass

            'context': self.context,class TaskResult:

            'constraints': self.constraints or {},    """Result of a glycoinformatics task"""

            'mode': self.mode.value    task_id: str

        }    status: TaskStatus

    results: Dict[str, Any]

    execution_time: float

@dataclass    component_results: Dict[str, Any] = field(default_factory=dict)

class ReasoningResponse:    errors: List[str] = field(default_factory=list)

    """Response from Graph of Thoughts reasoning"""    warnings: List[str] = field(default_factory=list)

    task_id: str    metadata: Dict[str, Any] = field(default_factory=dict)

    success: bool

    reasoning_paths: List[List[str]]

    best_path: List[str]@dataclass

    confidence: floatclass WorkflowStep:

    explanation: str    """A single step in a workflow"""

    graph_stats: Dict[str, Any]    step_id: str

    evaluation_metrics: Dict[str, Any]    component_type: ComponentType

    execution_time: float    operation: str

    metadata: Dict[str, Any]    inputs: Dict[str, Any]

        outputs: List[str]

    def to_dict(self) -> Dict[str, Any]:    dependencies: List[str] = field(default_factory=list)

        return {    parameters: Dict[str, Any] = field(default_factory=dict)

            'task_id': self.task_id,    optional: bool = False

            'success': self.success,

            'reasoning_paths': self.reasoning_paths,

            'best_path': self.best_path,@dataclass

            'confidence': self.confidence,class Workflow:

            'explanation': self.explanation,    """A complete workflow definition"""

            'graph_stats': self.graph_stats,    workflow_id: str

            'evaluation_metrics': self.evaluation_metrics,    name: str

            'execution_time': self.execution_time,    description: str

            'metadata': self.metadata    steps: List[WorkflowStep]

        }    task_type: TaskType

    expected_duration: float = 60.0

    metadata: Dict[str, Any] = field(default_factory=dict)

class GlycoGoTIntegrator:

    """

    Main integration class for Graph of Thoughts in glycoinformatics.class ComponentInterface(ABC):

        """Abstract interface for system components"""

    This class orchestrates the complete reasoning pipeline, integrating    

    with various platform components as needed.    @abstractmethod

    """    def initialize(self, config: Dict[str, Any]) -> bool:

            """Initialize the component"""

    def __init__(self,         pass

                 integration_mode: IntegrationMode = IntegrationMode.STANDALONE,        

                 max_thoughts: int = 1000,    @abstractmethod

                 max_reasoning_time: float = 300.0,    def process(self, 

                 llm_client=None,                operation: str,

                 knowledge_graph=None):                inputs: Dict[str, Any],

        """                parameters: Dict[str, Any] = None) -> Dict[str, Any]:

        Initialize the GoT integrator.        """Process data through the component"""

                pass

        Args:        

            integration_mode: How to integrate with the platform    @abstractmethod

            max_thoughts: Maximum thoughts to generate    def get_status(self) -> Dict[str, Any]:

            max_reasoning_time: Maximum reasoning time in seconds        """Get component status"""

            llm_client: Optional LLM client for enhanced reasoning        pass

            knowledge_graph: Optional knowledge graph for context        

        """    @abstractmethod

        self.integration_mode = integration_mode    def cleanup(self):

        self.max_thoughts = max_thoughts        """Cleanup component resources"""

        self.max_reasoning_time = max_reasoning_time        pass

        

        # Platform integrations

        self.llm_client = llm_clientclass KnowledgeGraphComponent(ComponentInterface):

        self.knowledge_graph = knowledge_graph    """Knowledge graph component wrapper"""

            

        # Core components    def __init__(self):

        self.got_engine = GraphOfThoughts(max_thoughts=max_thoughts)        self.ontology = None

        self.thought_generators = create_standard_generators()        self.query_manager = None

        self.path_evaluator = PathEvaluator()        self.glytoucan_client = None

        self.graph_evaluator = GraphEvaluator()        self.is_initialized = False

        self.comparator = ReasoningComparator()        

            def initialize(self, config: Dict[str, Any]) -> bool:

        # Task-specific configurations        """Initialize knowledge graph components"""

        self.task_configs = self._initialize_task_configs()        try:

                    # Import components

        # Performance tracking            from ..glycokg.ontology.glyco_ontology import GlycoOntology

        self.reasoning_sessions = []            from ..glycokg.query.sparql_utils import SPARQLQueryManager

                    from ..glycokg.integration.glytoucan_client import GlyTouCanClient

    def _initialize_task_configs(self) -> Dict[ReasoningTask, Dict[str, Any]]:            

        """Initialize task-specific configurations"""            # Initialize ontology

        return {            self.ontology = GlycoOntology()

            ReasoningTask.STRUCTURE_ELUCIDATION: {            

                'preferred_generators': ['deductive_reasoning', 'hypothesis', 'analogical_reasoning'],            # Initialize query manager

                'search_strategy': SearchStrategy.BEST_FIRST,            self.query_manager = SPARQLQueryManager(

                'confidence_threshold': 0.7,                local_graph=self.ontology.graph,

                'max_paths': 5,                cache_enabled=config.get('enable_cache', True)

                'reasoning_depth': 10            )

            },            

            ReasoningTask.PATHWAY_ANALYSIS: {            # Initialize GlyTouCan client

                'preferred_generators': ['causal_reasoning', 'synthesis', 'inductive_reasoning'],            self.glytoucan_client = GlyTouCanClient()

                'search_strategy': SearchStrategy.BREADTH_FIRST,            

                'confidence_threshold': 0.6,            self.is_initialized = True

                'max_paths': 8,            logger.info("Knowledge graph component initialized successfully")

                'reasoning_depth': 15            return True

            },            

            ReasoningTask.FUNCTION_PREDICTION: {        except Exception as e:

                'preferred_generators': ['analogical_reasoning', 'inductive_reasoning', 'hypothesis'],            logger.error(f"Failed to initialize knowledge graph component: {e}")

                'search_strategy': SearchStrategy.MONTE_CARLO,            return False

                'confidence_threshold': 0.5,            

                'max_paths': 10,    def process(self, 

                'reasoning_depth': 12                operation: str,

            },                inputs: Dict[str, Any],

            ReasoningTask.BIOMARKER_DISCOVERY: {                parameters: Dict[str, Any] = None) -> Dict[str, Any]:

                'preferred_generators': ['inductive_reasoning', 'synthesis', 'evaluation'],        """Process knowledge graph operations"""

                'search_strategy': SearchStrategy.A_STAR,        

                'confidence_threshold': 0.8,        if not self.is_initialized:

                'max_paths': 3,            return {"error": "Component not initialized"}

                'reasoning_depth': 8            

            }        parameters = parameters or {}

        }        

            try:

    async def process_reasoning_request(self, request: ReasoningRequest) -> ReasoningResponse:            if operation == "sparql_query":

        """                query = inputs.get("query", "")

        Process a complete reasoning request.                result = self.query_manager.execute_query(query)

                        return {

        Args:                    "results": result.results,

            request: The reasoning request to process                    "result_count": result.result_count,

                                "execution_time": result.execution_time,

        Returns:                    "error": result.error

            Complete reasoning response                }

        """                

        start_time = time.time()            elif operation == "glycan_lookup":

                        glycan_id = inputs.get("glycan_id", "")

        try:                if self.glytoucan_client:

            # Reset reasoning engine                    glycan_info = self.glytoucan_client.get_glycan_info(glycan_id)

            self.got_engine.reset_reasoning()                    return {"glycan_info": glycan_info}

                            else:

            # Set up reasoning goal                    return {"error": "GlyTouCan client not available"}

            goal_id = self.got_engine.set_goal(request.goal, request.context)                    

                        elif operation == "add_glycan":

            # Extract and add initial observations                glycan_data = inputs.get("glycan_data", {})

            observations = await self._extract_observations(request)                glycan_id = glycan_data.get("id", "")

            observation_ids = self.got_engine.add_initial_observations(observations)                wurcs = glycan_data.get("wurcs", "")

                            mass = glycan_data.get("mass", 0.0)

            # Configure task-specific settings                

            task_config = self.task_configs.get(request.task_type, {})                self.ontology.add_glycan(glycan_id, wurcs_sequence=wurcs, mass_mono=mass)

                            return {"status": "added", "glycan_id": glycan_id}

            # Generate thoughts iteratively                

            await self._iterative_thought_generation(            elif operation == "search_glycans":

                request, observation_ids, task_config                search_params = inputs.get("search_params", {})

            )                # Implement glycan search logic

                            query = self._build_search_query(search_params)

            # Search for solution paths                result = self.query_manager.execute_query(query)

            search_strategy = task_config.get('search_strategy', SearchStrategy.BEST_FIRST)                return {

            max_paths = task_config.get('max_paths', 5)                    "matching_glycans": result.results,

                                "count": result.result_count

            solution_paths = self.got_engine.search_solution_paths(                }

                strategy=search_strategy,                

                max_paths=max_paths            else:

            )                return {"error": f"Unknown operation: {operation}"}

                            

            # Evaluate solutions        except Exception as e:

            evaluations = self.got_engine.evaluate_solutions(solution_paths)            return {"error": f"Knowledge graph operation failed: {str(e)}"}

                        

            # Select best solution    def _build_search_query(self, search_params: Dict[str, Any]) -> str:

            best_evaluation = evaluations[0] if evaluations else None        """Build SPARQL query for glycan search"""

            best_path = best_evaluation['path'] if best_evaluation else []        

                    base_query = f"""

            # Generate explanation        PREFIX glyco: <{self.ontology.GLYCO}>

            explanation = self.got_engine.explain_reasoning(best_path) if best_path else "No solution found"        SELECT ?glycan ?id ?mass ?wurcs WHERE {{

                        ?glycan a glyco:Glycan .

            # Calculate confidence            ?glycan glyco:hasGlyTouCanID ?id .

            confidence = best_evaluation['overall_score'] if best_evaluation else 0.0        """

                    

            # Gather statistics        conditions = []

            graph_stats = self.got_engine.get_reasoning_statistics()        

            graph_eval = self.graph_evaluator.evaluate_graph(self.got_engine.graph)        if "min_mass" in search_params:

                        conditions.append(f"?glycan glyco:hasMonoisotopicMass ?mass . FILTER(?mass >= {search_params['min_mass']})")

            execution_time = time.time() - start_time            

                    if "max_mass" in search_params:

            # Create response            conditions.append(f"?glycan glyco:hasMonoisotopicMass ?mass . FILTER(?mass <= {search_params['max_mass']})")

            response = ReasoningResponse(            

                task_id=request.task_id,        if "monosaccharide" in search_params:

                success=len(best_path) > 0,            mono = search_params["monosaccharide"]

                reasoning_paths=solution_paths,            conditions.append(f"?glycan glyco:hasWURCS ?wurcs . FILTER(CONTAINS(LCASE(?wurcs), LCASE('{mono}')))")

                best_path=best_path,            

                confidence=confidence,        if conditions:

                explanation=explanation,            base_query += " " + " ".join(conditions)

                graph_stats=graph_stats,            

                evaluation_metrics=graph_eval.to_dict(),        base_query += " }"

                execution_time=execution_time,        

                metadata={        return base_query

                    'integration_mode': self.integration_mode.value,        

                    'task_type': request.task_type.value,    def get_status(self) -> Dict[str, Any]:

                    'thoughts_generated': len(self.got_engine.graph.nodes),        """Get knowledge graph component status"""

                    'search_strategy': search_strategy.value        status = {

                }            "initialized": self.is_initialized,

            )            "ontology_loaded": self.ontology is not None,

                        "query_manager_ready": self.query_manager is not None,

            # Track session            "glytoucan_client_ready": self.glytoucan_client is not None

            self.reasoning_sessions.append({        }

                'request': request.to_dict(),        

                'response': response.to_dict(),        if self.ontology:

                'timestamp': time.time()            status["ontology_triples"] = len(self.ontology.graph)

            })            

                    return status

            return response        

                def cleanup(self):

        except Exception as e:        """Cleanup knowledge graph resources"""

            logger.error(f"Error processing reasoning request: {e}")        self.ontology = None

            execution_time = time.time() - start_time        self.query_manager = None

                    self.glytoucan_client = None

            return ReasoningResponse(        self.is_initialized = False

                task_id=request.task_id,

                success=False,

                reasoning_paths=[],class AIModelComponent(ComponentInterface):

                best_path=[],    """AI model component wrapper"""

                confidence=0.0,    

                explanation=f"Error in reasoning: {str(e)}",    def __init__(self):

                graph_stats={},        self.tokenizer = None

                evaluation_metrics={},        self.model = None

                execution_time=execution_time,        self.dataset_builder = None

                metadata={'error': str(e)}        self.is_initialized = False

            )        

        def initialize(self, config: Dict[str, Any]) -> bool:

    async def _extract_observations(self, request: ReasoningRequest) -> List[str]:        """Initialize AI model components"""

        """Extract observations from input data based on task type"""        try:

        observations = []            # Import AI components

        input_data = request.input_data            from ..glycollm.tokenization.glyco_tokenizer import GlycoTokenizer

                    from ..glycollm.models.glycollm import GlycoLLM, GlycoLLMConfig

        if request.task_type == ReasoningTask.STRUCTURE_ELUCIDATION:            from ..glycollm.data.multimodal_dataset import MultimodalDatasetBuilder

            # Extract mass spec and structural observations            

            if 'mass_spectrum' in input_data:            # Initialize tokenizer

                spectrum = input_data['mass_spectrum']            self.tokenizer = GlycoTokenizer()

                if 'precursor_mass' in spectrum:            

                    observations.append(f"Precursor ion at m/z {spectrum['precursor_mass']}")            # Initialize model with config

                            model_config = GlycoLLMConfig(

                if 'fragments' in spectrum:                vocab_size=config.get('vocab_size', 50000),

                    for fragment in spectrum['fragments'][:5]:  # Limit to top 5                d_model=config.get('d_model', 768),

                        observations.append(f"Fragment ion at m/z {fragment.get('mz', 0)} with intensity {fragment.get('intensity', 0)}")                n_layers=config.get('n_layers', 12),

                            n_heads=config.get('n_heads', 12)

            if 'structural_features' in input_data:            )

                for feature in input_data['structural_features']:            

                    observations.append(f"Structural feature: {feature}")            try:

                                    self.model = GlycoLLM(model_config)

        elif request.task_type == ReasoningTask.PATHWAY_ANALYSIS:                logger.info("GlycoLLM model initialized")

            # Extract pathway and enzyme observations            except Exception as e:

            if 'enzymes' in input_data:                logger.warning(f"Could not initialize full model: {e}")

                for enzyme in input_data['enzymes']:                self.model = None

                    observations.append(f"Enzyme present: {enzyme}")                

                        # Initialize dataset builder

            if 'metabolites' in input_data:            self.dataset_builder = MultimodalDatasetBuilder()

                for metabolite in input_data['metabolites']:            

                    observations.append(f"Metabolite detected: {metabolite}")            self.is_initialized = True

                                logger.info("AI model component initialized successfully")

        elif request.task_type == ReasoningTask.FUNCTION_PREDICTION:            return True

            # Extract functional observations            

            if 'binding_data' in input_data:        except Exception as e:

                for binding in input_data['binding_data']:            logger.error(f"Failed to initialize AI model component: {e}")

                    observations.append(f"Binding interaction: {binding}")            return False

                        

            if 'expression_data' in input_data:    def process(self, 

                for expr in input_data['expression_data']:                operation: str,

                    observations.append(f"Expression pattern: {expr}")                inputs: Dict[str, Any],

                        parameters: Dict[str, Any] = None) -> Dict[str, Any]:

        # Add knowledge graph context if available        """Process AI model operations"""

        if self.integration_mode in [IntegrationMode.KNOWLEDGE_ENHANCED, IntegrationMode.FULL_PIPELINE]:        

            kg_observations = await self._extract_knowledge_graph_context(request)        if not self.is_initialized:

            observations.extend(kg_observations)            return {"error": "Component not initialized"}

                    

        # Ensure we have at least one observation        parameters = parameters or {}

        if not observations:        

            observations.append(f"Analysis task: {request.goal}")        try:

                    if operation == "tokenize":

        return observations                text = inputs.get("text", "")

                    wurcs = inputs.get("wurcs", "")

    async def _extract_knowledge_graph_context(self, request: ReasoningRequest) -> List[str]:                spectra = inputs.get("spectra", [])

        """Extract relevant context from knowledge graph"""                

        context_observations = []                result = {}

                        

        if self.knowledge_graph:                if text and hasattr(self.tokenizer, 'encode_text'):

            try:                    result["text_tokens"] = self.tokenizer.encode_text(text)

                # Query knowledge graph for relevant information                    

                # This would integrate with the actual knowledge graph                if wurcs and hasattr(self.tokenizer, 'encode_wurcs'):

                context_observations.append("Knowledge graph context: Related glycan structures found")                    result["wurcs_tokens"] = self.tokenizer.encode_wurcs(wurcs)

                context_observations.append("Knowledge graph context: Similar analysis patterns identified")                    

            except Exception as e:                if spectra and hasattr(self.tokenizer, 'encode_spectra'):

                logger.warning(f"Knowledge graph context extraction failed: {e}")                    result["spectra_tokens"] = self.tokenizer.encode_spectra(spectra)

                            

        return context_observations                return result

                    

    async def _iterative_thought_generation(self,             elif operation == "predict_structure":

                                          request: ReasoningRequest,                input_data = inputs.get("input_data", {})

                                          start_thought_ids: List[str],                 

                                          task_config: Dict[str, Any]):                if self.model is None:

        """Generate thoughts iteratively with task-specific focus"""                    return {"error": "Model not available"}

        current_thoughts = start_thought_ids                    

        max_iterations = task_config.get('reasoning_depth', 10)                # Simplified structure prediction

        preferred_generators = task_config.get('preferred_generators', [])                return {

                            "predicted_structure": "WURCS=predicted_structure",

        # Filter generators based on task preferences                    "confidence": 0.85,

        if preferred_generators:                    "note": "This is a mock prediction - full model inference would be implemented here"

            active_generators = [                }

                gen for gen in self.thought_generators                 

                if gen.get_generator_type() in preferred_generators            elif operation == "generate_text":

            ]                prompt = inputs.get("prompt", "")

        else:                max_length = parameters.get("max_length", 100)

            active_generators = self.thought_generators                

                        if self.model is None:

        generator_functions = generators_to_functions(active_generators)                    return {"error": "Model not available"}

                            

        for iteration in range(max_iterations):                # Mock text generation

            if len(self.got_engine.graph.nodes) >= self.max_thoughts:                return {

                break                    "generated_text": f"Generated response for: {prompt[:50]}...",

                                "note": "This is a mock generation - full model would be used here"

            # Check time limit                }

            if hasattr(self, '_reasoning_start_time'):                

                elapsed = time.time() - self._reasoning_start_time            elif operation == "build_dataset":

                if elapsed > self.max_reasoning_time:                data_sources = inputs.get("data_sources", {})

                    break                dataset_config = parameters.get("dataset_config", {})

                            

            # Generate new thoughts                # Use dataset builder to process data

            new_thoughts = self.got_engine.generate_thoughts(                dataset_info = self.dataset_builder.build_dataset(

                source_thoughts=current_thoughts,                    text_sources=data_sources.get("text_sources", []),

                thought_generators=generator_functions,                    spectra_sources=data_sources.get("spectra_sources", []),

                max_new_thoughts=min(20, self.max_thoughts - len(self.got_engine.graph.nodes))                    structure_sources=data_sources.get("structure_sources", [])

            )                )

                            

            if not new_thoughts:                return {

                break  # No more thoughts generated                    "dataset_info": dataset_info,

                                "status": "built"

            # LLM enhancement if available                }

            if self.integration_mode in [IntegrationMode.LLM_ASSISTED, IntegrationMode.FULL_PIPELINE]:                

                enhanced_thoughts = await self._enhance_thoughts_with_llm(new_thoughts, request)            else:

                new_thoughts.extend(enhanced_thoughts)                return {"error": f"Unknown operation: {operation}"}

                            

            # Filter high-quality thoughts for next iteration        except Exception as e:

            confidence_threshold = task_config.get('confidence_threshold', 0.5)            return {"error": f"AI model operation failed: {str(e)}"}

            current_thoughts = [            

                thought_id for thought_id in new_thoughts    def get_status(self) -> Dict[str, Any]:

                if self.got_engine.graph.get_thought(thought_id) and         """Get AI model component status"""

                   self.got_engine.graph.get_thought(thought_id).confidence >= confidence_threshold        return {

            ]            "initialized": self.is_initialized,

                        "tokenizer_ready": self.tokenizer is not None,

            if not current_thoughts:            "model_ready": self.model is not None,

                current_thoughts = new_thoughts[:5]  # Take top 5 if none meet threshold            "dataset_builder_ready": self.dataset_builder is not None

            }

    async def _enhance_thoughts_with_llm(self,         

                                       thought_ids: List[str],     def cleanup(self):

                                       request: ReasoningRequest) -> List[str]:        """Cleanup AI model resources"""

        """Enhance thoughts using LLM if available"""        self.tokenizer = None

        enhanced_thought_ids = []        self.model = None

                self.dataset_builder = None

        if not self.llm_client:        self.is_initialized = False

            return enhanced_thought_ids

        

        try:class ReasoningEngineComponent(ComponentInterface):

            for thought_id in thought_ids[:3]:  # Limit LLM calls    """Reasoning engine component wrapper"""

                thought = self.got_engine.graph.get_thought(thought_id)    

                if not thought:    def __init__(self):

                    continue        self.reasoner = None

                        self.is_initialized = False

                # Create LLM prompt for enhancement        

                prompt = self._create_llm_enhancement_prompt(thought, request)    def initialize(self, config: Dict[str, Any]) -> bool:

                        """Initialize reasoning engine"""

                # Get LLM response (this would call actual LLM)        try:

                # llm_response = await self.llm_client.generate(prompt)            from .reasoning import GlycoGOTReasoner

                llm_response = f"LLM enhancement of: {thought.content[:50]}..."            

                            self.reasoner = GlycoGOTReasoner()

                # Create enhanced thought            self.is_initialized = True

                enhanced_thought = ThoughtNode(            logger.info("Reasoning engine component initialized successfully")

                    content=llm_response,            return True

                    thought_type=ThoughtType.SYNTHESIS,            

                    confidence=thought.confidence * 0.8,  # Slightly lower confidence        except Exception as e:

                    relevance=thought.relevance,            logger.error(f"Failed to initialize reasoning engine component: {e}")

                    novelty=0.7,            return False

                    evidence=[thought.content],            

                    creator="llm_enhancer",    def process(self, 

                    context={'source_thought': thought_id, 'enhancement': 'llm'}                operation: str,

                )                inputs: Dict[str, Any],

                                parameters: Dict[str, Any] = None) -> Dict[str, Any]:

                if self.got_engine.graph.add_thought(enhanced_thought):        """Process reasoning operations"""

                    enhanced_thought_ids.append(enhanced_thought.id)        

                            if not self.is_initialized:

        except Exception as e:            return {"error": "Component not initialized"}

            logger.warning(f"LLM enhancement failed: {e}")            

                parameters = parameters or {}

        return enhanced_thought_ids        

            try:

    def _create_llm_enhancement_prompt(self, thought: ThoughtNode, request: ReasoningRequest) -> str:            if operation == "reason":

        """Create prompt for LLM enhancement"""                goal = inputs.get("goal", "")

        return f"""                context = inputs.get("context", {})

        Task: {request.task_type.value}                

        Goal: {request.goal}                result = self.reasoner.reason(goal, context)

        Current thought: {thought.content}                

                        return {

        Please enhance this thought with additional insights, details, or related information                    "reasoning_result": {

        that would be valuable for {request.task_type.value} in glycoinformatics.                        "goal": result.goal,

                                "conclusion": result.conclusion,

        Enhanced thought:                        "confidence": result.confidence,

        """                        "steps": len(result.reasoning_chain),

                            "execution_time": result.execution_time,

    def create_structure_elucidation_request(self,                         "success": result.success

                                           mass_spectrum_data: Dict[str, Any],                    },

                                           additional_context: Dict[str, Any] = None) -> ReasoningRequest:                    "explanation": self.reasoner.get_reasoning_explanation(result),

        """Create a structure elucidation reasoning request"""                    "alternative_hypotheses": result.alternative_hypotheses

        return ReasoningRequest(                }

            task_id=str(uuid.uuid4()),                

            task_type=ReasoningTask.STRUCTURE_ELUCIDATION,            elif operation == "generate_hypotheses":

            input_data={'mass_spectrum': mass_spectrum_data},                context = inputs.get("context", {})

            goal="Determine the glycan structure from mass spectrometry data",                max_hypotheses = parameters.get("max_hypotheses", 5)

            context=additional_context or {},                

            mode=self.integration_mode                hypotheses = self.reasoner.hypothesis_generator.generate_hypotheses(

        )                    context, max_hypotheses

                    )

    def create_pathway_analysis_request(self,                

                                      enzyme_data: List[str],                return {"hypotheses": hypotheses}

                                      metabolite_data: List[str],                

                                      pathway_goal: str) -> ReasoningRequest:            elif operation == "analyze_causality":

        """Create a pathway analysis reasoning request"""                observations = inputs.get("observations", [])

        return ReasoningRequest(                target_effect = inputs.get("target_effect", "")

            task_id=str(uuid.uuid4()),                

            task_type=ReasoningTask.PATHWAY_ANALYSIS,                causal_analysis = self.reasoner.causal_reasoner.analyze_causality(

            input_data={'enzymes': enzyme_data, 'metabolites': metabolite_data},                    observations, target_effect

            goal=pathway_goal,                )

            context={'analysis_type': 'pathway'},                

            mode=self.integration_mode                return {"causal_analysis": causal_analysis}

        )                

                else:

    def get_reasoning_statistics(self) -> Dict[str, Any]:                return {"error": f"Unknown operation: {operation}"}

        """Get comprehensive reasoning statistics"""                

        return {        except Exception as e:

            'total_sessions': len(self.reasoning_sessions),            return {"error": f"Reasoning operation failed: {str(e)}"}

            'integration_mode': self.integration_mode.value,            

            'success_rate': self._calculate_success_rate(),    def get_status(self) -> Dict[str, Any]:

            'average_execution_time': self._calculate_average_execution_time(),        """Get reasoning engine status"""

            'thought_generation_stats': self._calculate_thought_stats(),        status = {

            'recent_sessions': self.reasoning_sessions[-5:] if self.reasoning_sessions else []            "initialized": self.is_initialized,

        }            "reasoner_ready": self.reasoner is not None

            }

    def _calculate_success_rate(self) -> float:        

        """Calculate success rate across sessions"""        if self.reasoner:

        if not self.reasoning_sessions:            status["reasoning_history_count"] = len(self.reasoner.reasoning_history)

            return 0.0            

                return status

        successful = sum(1 for session in self.reasoning_sessions         

                        if session['response']['success'])    def cleanup(self):

        return successful / len(self.reasoning_sessions)        """Cleanup reasoning engine resources"""

            self.reasoner = None

    def _calculate_average_execution_time(self) -> float:        self.is_initialized = False

        """Calculate average execution time"""

        if not self.reasoning_sessions:

            return 0.0class GlycoGOTIntegrator:

            """Main integration coordinator for GlycoGOT system"""

        times = [session['response']['execution_time'] for session in self.reasoning_sessions]    

        return sum(times) / len(times)    def __init__(self, config: Dict[str, Any] = None):

            self.config = config or {}

    def _calculate_thought_stats(self) -> Dict[str, Any]:        

        """Calculate thought generation statistics"""        # Initialize components

        if not self.reasoning_sessions:        self.components = {

            return {}            ComponentType.KNOWLEDGE_GRAPH: KnowledgeGraphComponent(),

                    ComponentType.AI_MODEL: AIModelComponent(),

        thought_counts = [session['response']['metadata'].get('thoughts_generated', 0)             ComponentType.REASONING_ENGINE: ReasoningEngineComponent()

                         for session in self.reasoning_sessions]        }

                

        return {        # Task management

            'avg_thoughts_per_session': sum(thought_counts) / len(thought_counts) if thought_counts else 0,        self.active_tasks = {}

            'max_thoughts_generated': max(thought_counts) if thought_counts else 0,        self.completed_tasks = {}

            'min_thoughts_generated': min(thought_counts) if thought_counts else 0        self.workflows = {}

        }        

            # Execution pool for concurrent tasks

    async def batch_process_requests(self, requests: List[ReasoningRequest]) -> List[ReasoningResponse]:        self.executor = ThreadPoolExecutor(max_workers=4) if HAS_CONCURRENT else None

        """Process multiple reasoning requests in batch"""        

        responses = []        # Initialize predefined workflows

                self._initialize_workflows()

        for request in requests:        

            response = await self.process_reasoning_request(request)        logger.info("GlycoGOT Integrator initialized")

            responses.append(response)        

            def initialize_components(self) -> Dict[ComponentType, bool]:

        return responses        """Initialize all system components"""

            initialization_results = {}

    def export_reasoning_session(self, session_index: int = -1) -> Dict[str, Any]:        

        """Export a reasoning session for analysis"""        for component_type, component in self.components.items():

        if not self.reasoning_sessions:            component_config = self.config.get(component_type.value, {})

            return {}            success = component.initialize(component_config)

                    initialization_results[component_type] = success

        session = self.reasoning_sessions[session_index]            

                    if success:

        # Add graph export                logger.info(f"{component_type.value} initialized successfully")

        session_export = session.copy()            else:

        session_export['graph_export'] = self.got_engine.graph.to_dict()                logger.error(f"Failed to initialize {component_type.value}")

                        

        return session_export        return initialization_results

        

    def _initialize_workflows(self):

# Utility functions for integration        """Initialize predefined workflows"""

def create_mass_spec_reasoning_request(precursor_mass: float,        

                                     fragments: List[Dict[str, float]],        # Structure identification workflow

                                     sample_info: Dict[str, Any] = None) -> ReasoningRequest:        self.workflows["structure_identification"] = Workflow(

    """Create a reasoning request for mass spec structure elucidation"""            workflow_id="struct_id_001",

    integrator = GlycoGoTIntegrator()            name="Glycan Structure Identification",

                description="Identify glycan structure from mass spectral data",

    mass_spectrum_data = {            task_type=TaskType.STRUCTURE_IDENTIFICATION,

        'precursor_mass': precursor_mass,            steps=[

        'fragments': fragments,                WorkflowStep(

        'sample_info': sample_info or {}                    step_id="preprocess_spectra",

    }                    component_type=ComponentType.AI_MODEL,

                        operation="tokenize",

    return integrator.create_structure_elucidation_request(mass_spectrum_data)                    inputs={"spectra": "input_spectra"},

                    outputs=["preprocessed_spectra"]

                ),

async def quick_structure_analysis(precursor_mass: float,                 WorkflowStep(

                                 fragments: List[Dict[str, float]]) -> Dict[str, Any]:                    step_id="search_database",

    """Quick structure analysis using Graph of Thoughts"""                    component_type=ComponentType.KNOWLEDGE_GRAPH,

    integrator = GlycoGoTIntegrator(integration_mode=IntegrationMode.STANDALONE)                    operation="search_glycans",

                        inputs={"search_params": "mass_range"},

    request = create_mass_spec_reasoning_request(precursor_mass, fragments)                    outputs=["candidate_structures"],

    response = await integrator.process_reasoning_request(request)                    dependencies=["preprocess_spectra"]

                    ),

    return {                WorkflowStep(

        'success': response.success,                    step_id="predict_structure",

        'confidence': response.confidence,                    component_type=ComponentType.AI_MODEL,

        'explanation': response.explanation,                    operation="predict_structure",

        'execution_time': response.execution_time                    inputs={"input_data": "preprocessed_spectra"},

    }                    outputs=["predicted_structure"],

                    dependencies=["preprocess_spectra"]

                ),

# Example usage and testing                WorkflowStep(

if __name__ == "__main__":                    step_id="reason_about_structure",

    async def test_integration():                    component_type=ComponentType.REASONING_ENGINE,

        """Test the integration functionality"""                    operation="reason",

        integrator = GlycoGoTIntegrator(integration_mode=IntegrationMode.STANDALONE)                    inputs={"goal": "identify_most_likely_structure", "context": "all_data"},

                            outputs=["reasoning_result"],

        # Test structure elucidation                    dependencies=["search_database", "predict_structure"]

        mass_spec_data = {                )

            'precursor_mass': 1234.56,            ]

            'fragments': [        )

                {'mz': 666.33, 'intensity': 100},        

                {'mz': 504.22, 'intensity': 80},        # Function prediction workflow

                {'mz': 204.09, 'intensity': 90}        self.workflows["function_prediction"] = Workflow(

            ]            workflow_id="func_pred_001",

        }            name="Glycan Function Prediction",

                    description="Predict biological function of glycan structure",

        request = integrator.create_structure_elucidation_request(mass_spec_data)            task_type=TaskType.FUNCTION_PREDICTION,

        response = await integrator.process_reasoning_request(request)            steps=[

                        WorkflowStep(

        print(f"Analysis success: {response.success}")                    step_id="analyze_structure",

        print(f"Confidence: {response.confidence:.3f}")                    component_type=ComponentType.KNOWLEDGE_GRAPH,

        print(f"Execution time: {response.execution_time:.3f}s")                    operation="glycan_lookup",

        print(f"Explanation: {response.explanation[:200]}...")                    inputs={"glycan_id": "input_glycan_id"},

                            outputs=["structural_info"]

        # Get statistics                ),

        stats = integrator.get_reasoning_statistics()                WorkflowStep(

        print(f"Integration statistics: {stats}")                    step_id="predict_function",

                        component_type=ComponentType.AI_MODEL,

    # Run test                    operation="predict_structure",  # Would be predict_function in real implementation

    asyncio.run(test_integration())                    inputs={"input_data": "structural_info"},
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