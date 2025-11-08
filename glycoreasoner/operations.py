"""
Operations module for GlycoGoT reasoning system.

This module provides operational capabilities including batch processing,
workflow management, and reasoning result storage/retrieval.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncIterator
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import uuid

from .integration import (
    ReasoningOrchestrator, GlycoGoTIntegrator, 
    ReasoningRequest, ReasoningResponse
)
from .reasoning import ReasoningType, ReasoningChain

logger = logging.getLogger(__name__)


@dataclass
class BatchReasoningJob:
    """Configuration for batch reasoning jobs."""
    
    job_id: str
    name: str
    input_glycans: List[Union[str, Dict[str, Any]]]
    reasoning_tasks: List[ReasoningType]
    analysis_depth: str = "standard"
    output_format: str = "json"  # json, csv, xlsx
    output_path: Optional[str] = None
    created_at: datetime = None
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    results: List[ReasoningResponse] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = []


@dataclass
class ReasoningWorkflow:
    """Multi-step reasoning workflow configuration."""
    
    workflow_id: str
    name: str
    steps: List[Dict[str, Any]]
    input_data: Dict[str, Any]
    dependencies: Dict[str, List[str]] = None  # step dependencies
    parallel_execution: bool = False
    created_at: datetime = None
    status: str = "pending"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = {}


class ReasoningResultStore:
    """
    Storage and retrieval system for reasoning results.
    
    Provides caching, persistence, and query capabilities for
    reasoning analyses.
    """
    
    def __init__(self, storage_path: str = "./reasoning_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for recent results
        self.cache = {}
        self.cache_size_limit = 1000
        
    async def store_result(self, response: ReasoningResponse) -> str:
        """
        Store reasoning result.
        
        Args:
            response: Reasoning response to store
            
        Returns:
            Storage key for retrieval
        """
        
        storage_key = f"{response.request_id}_{int(time.time())}"
        
        # Store to file
        result_file = self.storage_path / f"{storage_key}.json"
        
        try:
            result_data = {
                'storage_key': storage_key,
                'stored_at': datetime.now().isoformat(),
                'response': self._serialize_response(response)
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
                
            # Add to cache
            self.cache[storage_key] = response
            
            # Maintain cache size limit
            if len(self.cache) > self.cache_size_limit:
                # Remove oldest entries
                sorted_keys = sorted(self.cache.keys())
                for key in sorted_keys[:len(self.cache) - self.cache_size_limit]:
                    del self.cache[key]
                    
            logger.info(f"Stored reasoning result: {storage_key}")
            return storage_key
            
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            raise
            
    async def retrieve_result(self, storage_key: str) -> Optional[ReasoningResponse]:
        """
        Retrieve reasoning result by key.
        
        Args:
            storage_key: Storage key
            
        Returns:
            Reasoning response or None if not found
        """
        
        # Check cache first
        if storage_key in self.cache:
            return self.cache[storage_key]
            
        # Load from file
        result_file = self.storage_path / f"{storage_key}.json"
        
        if not result_file.exists():
            return None
            
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                
            response = self._deserialize_response(result_data['response'])
            
            # Add to cache
            self.cache[storage_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving result {storage_key}: {e}")
            return None
            
    async def search_results(self, 
                           criteria: Dict[str, Any],
                           limit: int = 100) -> List[Tuple[str, ReasoningResponse]]:
        """
        Search stored results by criteria.
        
        Args:
            criteria: Search criteria dictionary
            limit: Maximum results to return
            
        Returns:
            List of (storage_key, response) tuples
        """
        
        results = []
        
        # Search through stored files
        for result_file in self.storage_path.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    
                response = self._deserialize_response(result_data['response'])
                
                if self._matches_criteria(response, criteria):
                    storage_key = result_data['storage_key']
                    results.append((storage_key, response))
                    
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading result file {result_file}: {e}")
                continue
                
        return results
        
    def _serialize_response(self, response: ReasoningResponse) -> Dict[str, Any]:
        """Serialize response for storage."""
        
        serialized = asdict(response)
        
        # Convert reasoning chains
        serialized['reasoning_chains'] = [
            asdict(chain) for chain in response.reasoning_chains
        ]
        
        return serialized
        
    def _deserialize_response(self, data: Dict[str, Any]) -> ReasoningResponse:
        """Deserialize response from storage."""
        
        # Convert reasoning chains back
        chains = []
        for chain_data in data.get('reasoning_chains', []):
            # Convert steps
            from .reasoning import ReasoningStep
            steps = [ReasoningStep(**step_data) for step_data in chain_data.get('steps', [])]
            
            chain_data['steps'] = steps
            
            # Convert reasoning type
            reasoning_type_str = chain_data.get('reasoning_type')
            if isinstance(reasoning_type_str, str):
                chain_data['reasoning_type'] = ReasoningType(reasoning_type_str)
                
            chains.append(ReasoningChain(**chain_data))
            
        data['reasoning_chains'] = chains
        
        return ReasoningResponse(**data)
        
    def _matches_criteria(self, response: ReasoningResponse, criteria: Dict[str, Any]) -> bool:
        """Check if response matches search criteria."""
        
        # Status filter
        if 'status' in criteria and response.status != criteria['status']:
            return False
            
        # Confidence threshold
        if 'min_confidence' in criteria and response.confidence_score < criteria['min_confidence']:
            return False
            
        # Reasoning type filter
        if 'reasoning_types' in criteria:
            required_types = set(criteria['reasoning_types'])
            response_types = set(chain.reasoning_type for chain in response.reasoning_chains)
            if not required_types.intersection(response_types):
                return False
                
        # Structure filter (if glycan structure is in input)
        if 'structure_contains' in criteria:
            search_term = criteria['structure_contains'].lower()
            for chain in response.reasoning_chains:
                input_structure = chain.input_glycan.get('structure', '').lower()
                if search_term in input_structure:
                    return True
            return False
            
        return True


class BatchReasoningProcessor:
    """
    Processes batch reasoning jobs with progress tracking and error handling.
    
    Supports parallel processing and result aggregation.
    """
    
    def __init__(self, 
                 orchestrator: ReasoningOrchestrator,
                 result_store: ReasoningResultStore,
                 max_concurrent: int = 5):
        
        self.orchestrator = orchestrator
        self.result_store = result_store
        self.max_concurrent = max_concurrent
        
        # Active jobs
        self.active_jobs = {}
        
    async def submit_batch_job(self, job: BatchReasoningJob) -> str:
        """
        Submit batch reasoning job for processing.
        
        Args:
            job: Batch job configuration
            
        Returns:
            Job ID for tracking
        """
        
        job.status = "pending"
        job.progress = 0.0
        
        self.active_jobs[job.job_id] = job
        
        # Start processing in background
        asyncio.create_task(self._process_batch_job(job))
        
        logger.info(f"Submitted batch job: {job.job_id} with {len(job.input_glycans)} glycans")
        
        return job.job_id
        
    async def _process_batch_job(self, job: BatchReasoningJob):
        """Process batch job with progress tracking."""
        
        try:
            job.status = "running"
            
            total_glycans = len(job.input_glycans)
            completed = 0
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_single_glycan(glycan_input, index):
                async with semaphore:
                    try:
                        # Create reasoning request
                        request = ReasoningRequest(
                            request_id=f"{job.job_id}_glycan_{index}",
                            reasoning_tasks=job.reasoning_tasks
                        )
                        
                        # Set glycan data based on input type
                        if isinstance(glycan_input, str):
                            if glycan_input.startswith('WURCS'):
                                request.structure = glycan_input
                            else:
                                request.glycan_identifier = glycan_input
                        elif isinstance(glycan_input, dict):
                            request.structure = glycan_input.get('structure')
                            request.glycan_identifier = glycan_input.get('id')
                            request.spectra = glycan_input.get('spectra')
                            request.text_description = glycan_input.get('text')
                            request.context = glycan_input.get('context', {})
                            
                        # Process request
                        response = await self.orchestrator.integrator.process_reasoning_request(request)
                        
                        # Store result
                        storage_key = await self.result_store.store_result(response)
                        response.additional_data = response.additional_data or {}
                        response.additional_data['storage_key'] = storage_key
                        
                        return response
                        
                    except Exception as e:
                        logger.error(f"Error processing glycan {index} in job {job.job_id}: {e}")
                        
                        # Create error response
                        error_response = ReasoningResponse(
                            request_id=f"{job.job_id}_glycan_{index}",
                            status="error",
                            reasoning_chains=[],
                            summary=f"Processing error: {str(e)}",
                            confidence_score=0.0,
                            execution_time=0.0,
                            error_message=str(e)
                        )
                        
                        return error_response
                        
            # Process glycans concurrently
            tasks = [
                process_single_glycan(glycan, i) 
                for i, glycan in enumerate(job.input_glycans)
            ]
            
            # Process with progress updates
            for completed_task in asyncio.as_completed(tasks):
                result = await completed_task
                job.results.append(result)
                
                completed += 1
                job.progress = completed / total_glycans
                
                logger.info(f"Job {job.job_id} progress: {completed}/{total_glycans} ({job.progress:.1%})")
                
            # Generate output file if requested
            if job.output_path:
                await self._generate_output_file(job)
                
            job.status = "completed"
            job.progress = 1.0
            
            logger.info(f"Completed batch job: {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Batch job {job.job_id} failed: {e}")
            
    async def _generate_output_file(self, job: BatchReasoningJob):
        """Generate output file for batch job results."""
        
        output_path = Path(job.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if job.output_format == "json":
            await self._generate_json_output(job, output_path)
        elif job.output_format == "csv":
            await self._generate_csv_output(job, output_path)
        else:
            logger.warning(f"Unsupported output format: {job.output_format}")
            
    async def _generate_json_output(self, job: BatchReasoningJob, output_path: Path):
        """Generate JSON output file."""
        
        output_data = {
            'job_info': {
                'job_id': job.job_id,
                'name': job.name,
                'created_at': job.created_at.isoformat(),
                'total_glycans': len(job.input_glycans),
                'successful_analyses': len([r for r in job.results if r.status == 'success'])
            },
            'results': [
                self.result_store._serialize_response(result) 
                for result in job.results
            ]
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
            
    async def _generate_csv_output(self, job: BatchReasoningJob, output_path: Path):
        """Generate CSV output file."""
        
        import csv
        
        with open(output_path.with_suffix('.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Request_ID', 'Status', 'Confidence_Score', 'Execution_Time',
                'Summary', 'Reasoning_Tasks', 'Error_Message'
            ])
            
            # Results
            for result in job.results:
                reasoning_tasks = ', '.join([
                    chain.reasoning_type.value 
                    for chain in result.reasoning_chains
                ])
                
                writer.writerow([
                    result.request_id,
                    result.status,
                    f"{result.confidence_score:.3f}",
                    f"{result.execution_time:.2f}",
                    result.summary.replace('\n', ' ')[:100] + '...' if len(result.summary) > 100 else result.summary,
                    reasoning_tasks,
                    result.error_message or ''
                ])
                
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        
        if job_id not in self.active_jobs:
            return None
            
        job = self.active_jobs[job_id]
        
        return {
            'job_id': job.job_id,
            'name': job.name,
            'status': job.status,
            'progress': job.progress,
            'total_glycans': len(job.input_glycans),
            'completed_analyses': len(job.results),
            'successful_analyses': len([r for r in job.results if r.status == 'success']),
            'created_at': job.created_at.isoformat(),
            'output_path': job.output_path
        }
        
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel running batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        
        if job_id not in self.active_jobs:
            return False
            
        job = self.active_jobs[job_id]
        
        if job.status == "running":
            job.status = "cancelled"
            logger.info(f"Cancelled batch job: {job_id}")
            return True
            
        return False


class WorkflowManager:
    """
    Manages complex multi-step reasoning workflows.
    
    Supports step dependencies, parallel execution, and workflow templates.
    """
    
    def __init__(self, 
                 orchestrator: ReasoningOrchestrator,
                 result_store: ReasoningResultStore):
        
        self.orchestrator = orchestrator
        self.result_store = result_store
        
        # Active workflows
        self.active_workflows = {}
        
        # Workflow templates
        self.workflow_templates = self._create_default_templates()
        
    def _create_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create default workflow templates."""
        
        templates = {
            'comprehensive_analysis': {
                'name': 'Comprehensive Glycan Analysis',
                'description': 'Complete structural and functional analysis',
                'steps': [
                    {
                        'id': 'structure_analysis',
                        'type': 'reasoning',
                        'reasoning_type': ReasoningType.STRUCTURE_ANALYSIS.value,
                        'description': 'Analyze glycan structure and composition'
                    },
                    {
                        'id': 'fragmentation_prediction',
                        'type': 'reasoning', 
                        'reasoning_type': ReasoningType.FRAGMENTATION_PREDICTION.value,
                        'description': 'Predict mass spectrometry fragmentation',
                        'depends_on': ['structure_analysis']
                    },
                    {
                        'id': 'pathway_inference',
                        'type': 'reasoning',
                        'reasoning_type': ReasoningType.PATHWAY_INFERENCE.value,
                        'description': 'Infer biosynthetic pathway',
                        'depends_on': ['structure_analysis']
                    },
                    {
                        'id': 'function_prediction',
                        'type': 'reasoning',
                        'reasoning_type': ReasoningType.FUNCTION_PREDICTION.value,
                        'description': 'Predict biological functions',
                        'depends_on': ['structure_analysis', 'pathway_inference']
                    }
                ],
                'parallel_execution': True
            },
            
            'ms_analysis': {
                'name': 'Mass Spectrometry Analysis',
                'description': 'Focused analysis for MS data interpretation',
                'steps': [
                    {
                        'id': 'structure_analysis',
                        'type': 'reasoning',
                        'reasoning_type': ReasoningType.STRUCTURE_ANALYSIS.value,
                        'description': 'Analyze structure for MS interpretation'
                    },
                    {
                        'id': 'fragmentation_prediction',
                        'type': 'reasoning',
                        'reasoning_type': ReasoningType.FRAGMENTATION_PREDICTION.value,
                        'description': 'Predict fragmentation pattern',
                        'depends_on': ['structure_analysis']
                    }
                ],
                'parallel_execution': False
            }
        }
        
        return templates
        
    async def create_workflow_from_template(self, 
                                          template_name: str,
                                          input_data: Dict[str, Any],
                                          workflow_name: Optional[str] = None) -> str:
        """
        Create workflow from template.
        
        Args:
            template_name: Name of workflow template
            input_data: Input data for workflow
            workflow_name: Custom name for workflow instance
            
        Returns:
            Workflow ID
        """
        
        if template_name not in self.workflow_templates:
            raise ValueError(f"Unknown workflow template: {template_name}")
            
        template = self.workflow_templates[template_name]
        
        workflow_id = str(uuid.uuid4())
        
        workflow = ReasoningWorkflow(
            workflow_id=workflow_id,
            name=workflow_name or f"{template['name']} - {workflow_id[:8]}",
            steps=template['steps'].copy(),
            input_data=input_data,
            parallel_execution=template.get('parallel_execution', False)
        )
        
        # Build dependencies
        for step in workflow.steps:
            step_id = step['id']
            deps = step.get('depends_on', [])
            workflow.dependencies[step_id] = deps
            
        self.active_workflows[workflow_id] = workflow
        
        return workflow_id
        
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow execution results
        """
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
            
        workflow = self.active_workflows[workflow_id]
        workflow.status = "running"
        
        try:
            if workflow.parallel_execution:
                results = await self._execute_workflow_parallel(workflow)
            else:
                results = await self._execute_workflow_sequential(workflow)
                
            workflow.status = "completed"
            
            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'results': results
            }
            
        except Exception as e:
            workflow.status = "failed"
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e)
            }
            
    async def _execute_workflow_sequential(self, workflow: ReasoningWorkflow) -> Dict[str, Any]:
        """Execute workflow steps sequentially."""
        
        results = {}
        
        for step in workflow.steps:
            step_id = step['id']
            
            # Check dependencies
            for dep_id in workflow.dependencies.get(step_id, []):
                if dep_id not in results:
                    raise ValueError(f"Step {step_id} depends on {dep_id} which has not completed")
                    
            # Execute step
            step_result = await self._execute_workflow_step(step, workflow.input_data, results)
            results[step_id] = step_result
            
        return results
        
    async def _execute_workflow_parallel(self, workflow: ReasoningWorkflow) -> Dict[str, Any]:
        """Execute workflow steps with dependency-aware parallelization."""
        
        results = {}
        completed_steps = set()
        pending_steps = {step['id']: step for step in workflow.steps}
        
        while pending_steps:
            # Find steps that can run (dependencies satisfied)
            ready_steps = []
            
            for step_id, step in pending_steps.items():
                deps = workflow.dependencies.get(step_id, [])
                if all(dep in completed_steps for dep in deps):
                    ready_steps.append((step_id, step))
                    
            if not ready_steps:
                raise ValueError("Circular dependency detected in workflow")
                
            # Execute ready steps in parallel
            tasks = [
                self._execute_workflow_step(step, workflow.input_data, results)
                for step_id, step in ready_steps
            ]
            
            step_results = await asyncio.gather(*tasks)
            
            # Update results and completed steps
            for (step_id, step), result in zip(ready_steps, step_results):
                results[step_id] = result
                completed_steps.add(step_id)
                del pending_steps[step_id]
                
        return results
        
    async def _execute_workflow_step(self, 
                                   step: Dict[str, Any],
                                   input_data: Dict[str, Any],
                                   previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single workflow step."""
        
        step_type = step.get('type', 'reasoning')
        
        if step_type == 'reasoning':
            # Create reasoning request
            reasoning_type = ReasoningType(step['reasoning_type'])
            
            request = ReasoningRequest(
                request_id=f"workflow_{step['id']}_{int(time.time())}",
                structure=input_data.get('structure'),
                glycan_identifier=input_data.get('glycan_identifier'),
                spectra=input_data.get('spectra'),
                text_description=input_data.get('text_description'),
                reasoning_tasks=[reasoning_type],
                context={
                    'workflow_step': step['id'],
                    'previous_results': previous_results
                }
            )
            
            # Execute reasoning
            response = await self.orchestrator.integrator.process_reasoning_request(request)
            
            # Store result
            storage_key = await self.result_store.store_result(response)
            
            return {
                'step_id': step['id'],
                'type': step_type,
                'response': response,
                'storage_key': storage_key
            }
        else:
            raise ValueError(f"Unknown step type: {step_type}")


def create_operations_suite(orchestrator: ReasoningOrchestrator,
                          storage_path: str = "./reasoning_results") -> Dict[str, Any]:
    """
    Create complete operations suite for GlycoGoT.
    
    Args:
        orchestrator: Reasoning orchestrator
        storage_path: Path for result storage
        
    Returns:
        Dictionary with all operational components
    """
    
    # Create result store
    result_store = ReasoningResultStore(storage_path)
    
    # Create batch processor
    batch_processor = BatchReasoningProcessor(
        orchestrator=orchestrator,
        result_store=result_store
    )
    
    # Create workflow manager
    workflow_manager = WorkflowManager(
        orchestrator=orchestrator,
        result_store=result_store
    )
    
    return {
        'result_store': result_store,
        'batch_processor': batch_processor,
        'workflow_manager': workflow_manager,
        'orchestrator': orchestrator
    }