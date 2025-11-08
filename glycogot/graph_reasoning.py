"""
Graph-Based Reasoning Core for GoT Implementation
===============================================

This module implements the core graph structures and reasoning engine
for true Graph of Thoughts (GoT) in glycoinformatics.
"""

import uuid
import logging
import time
import heapq
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thoughts in the reasoning graph"""
    OBSERVATION = "observation"           # Initial observations/facts
    HYPOTHESIS = "hypothesis"             # Generated hypotheses
    DEDUCTION = "deduction"              # Logical deductions
    INDUCTION = "induction"              # Pattern-based inferences
    ANALOGY = "analogy"                  # Analogical reasoning
    CAUSAL = "causal"                    # Cause-effect relationships
    SYNTHESIS = "synthesis"              # Combining multiple thoughts
    EVALUATION = "evaluation"            # Evaluating other thoughts
    GOAL = "goal"                        # Goal/target thoughts


class ThoughtStatus(Enum):
    """Status of a thought node"""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class SearchStrategy(Enum):
    """Graph search strategies"""
    BREADTH_FIRST = "bfs"
    DEPTH_FIRST = "dfs"
    BEST_FIRST = "best_first"
    A_STAR = "a_star"
    MONTE_CARLO = "monte_carlo"


@dataclass
class ThoughtNode:
    """
    A single thought node in the reasoning graph.
    
    Each node represents one atomic piece of reasoning,
    with connections to other related thoughts.
    """
    
    # Core properties
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                           # The actual thought content
    thought_type: ThoughtType = ThoughtType.OBSERVATION
    
    # Graph relationships
    parents: Set[str] = field(default_factory=set)     # Nodes this depends on
    children: Set[str] = field(default_factory=set)    # Nodes that depend on this
    
    # Evaluation metrics
    confidence: float = 0.0                     # Confidence in this thought [0-1]
    relevance: float = 0.0                      # Relevance to current goal [0-1]
    novelty: float = 0.0                        # How novel/creative this thought is
    evidence_strength: float = 0.0              # Strength of supporting evidence
    
    # Metadata
    status: ThoughtStatus = ThoughtStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    creator: str = "system"                     # Who/what created this thought
    
    # Context and supporting information
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def add_parent(self, parent_id: str):
        """Add a parent dependency"""
        self.parents.add(parent_id)
        self.updated_at = time.time()
    
    def add_child(self, child_id: str):
        """Add a child dependency"""
        self.children.add(child_id)
        self.updated_at = time.time()
    
    def remove_parent(self, parent_id: str):
        """Remove a parent dependency"""
        self.parents.discard(parent_id)
        self.updated_at = time.time()
    
    def remove_child(self, child_id: str):
        """Remove a child dependency"""
        self.children.discard(child_id)
        self.updated_at = time.time()
    
    def update_status(self, status: ThoughtStatus):
        """Update the thought status"""
        self.status = status
        self.updated_at = time.time()
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall thought quality score"""
        if weights is None:
            weights = {
                'confidence': 0.3,
                'relevance': 0.3, 
                'evidence_strength': 0.25,
                'novelty': 0.15
            }
        
        return (
            self.confidence * weights.get('confidence', 0.3) +
            self.relevance * weights.get('relevance', 0.3) +
            self.evidence_strength * weights.get('evidence_strength', 0.25) +
            self.novelty * weights.get('novelty', 0.15)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'content': self.content,
            'thought_type': self.thought_type.value,
            'parents': list(self.parents),
            'children': list(self.children),
            'confidence': self.confidence,
            'relevance': self.relevance,
            'novelty': self.novelty,
            'evidence_strength': self.evidence_strength,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'creator': self.creator,
            'context': self.context,
            'evidence': self.evidence,
            'assumptions': self.assumptions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtNode':
        """Create ThoughtNode from dictionary"""
        node = cls(
            id=data.get('id', str(uuid.uuid4())),
            content=data.get('content', ''),
            thought_type=ThoughtType(data.get('thought_type', 'observation')),
            confidence=data.get('confidence', 0.0),
            relevance=data.get('relevance', 0.0),
            novelty=data.get('novelty', 0.0),
            evidence_strength=data.get('evidence_strength', 0.0),
            status=ThoughtStatus(data.get('status', 'created')),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            creator=data.get('creator', 'system'),
            context=data.get('context', {}),
            evidence=data.get('evidence', []),
            assumptions=data.get('assumptions', [])
        )
        
        node.parents = set(data.get('parents', []))
        node.children = set(data.get('children', []))
        
        return node


@dataclass 
class ThoughtEdge:
    """
    Represents a connection between two thoughts.
    
    Edges encode the relationships and dependencies between thoughts,
    including the strength and type of connection.
    """
    
    source_id: str                              # Source thought ID
    target_id: str                              # Target thought ID
    edge_type: str = "depends_on"               # Type of relationship
    weight: float = 1.0                         # Connection strength [0-1]
    confidence: float = 1.0                     # Confidence in this connection
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    creator: str = "system"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'weight': self.weight,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'creator': self.creator,
            'context': self.context
        }


class ThoughtGraph:
    """
    Core graph structure for storing and manipulating thoughts.
    
    This class manages the graph topology and provides efficient
    access to thoughts and their relationships.
    """
    
    def __init__(self, use_networkx: bool = True):
        """
        Initialize the thought graph.
        
        Args:
            use_networkx: Whether to use NetworkX for graph operations
        """
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: Dict[Tuple[str, str], ThoughtEdge] = {}
        
        # Optional NetworkX integration for advanced graph algorithms
        self.use_networkx = use_networkx and HAS_NETWORKX
        if self.use_networkx:
            self.nx_graph = nx.DiGraph()
            
        # Graph metadata
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata = {}
    
    def add_thought(self, thought: ThoughtNode) -> bool:
        """Add a thought node to the graph"""
        if thought.id in self.nodes:
            logger.warning(f"Thought {thought.id} already exists")
            return False
            
        self.nodes[thought.id] = thought
        
        if self.use_networkx:
            self.nx_graph.add_node(thought.id, **thought.to_dict())
            
        self.updated_at = time.time()
        return True
    
    def add_edge(self, edge: ThoughtEdge) -> bool:
        """Add an edge between two thoughts"""
        # Validate nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.error(f"Cannot add edge: nodes {edge.source_id} or {edge.target_id} not found")
            return False
            
        # Add to edge storage
        edge_key = (edge.source_id, edge.target_id)
        self.edges[edge_key] = edge
        
        # Update node relationships
        self.nodes[edge.source_id].add_child(edge.target_id)
        self.nodes[edge.target_id].add_parent(edge.source_id)
        
        if self.use_networkx:
            self.nx_graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
            
        self.updated_at = time.time()
        return True
    
    def remove_thought(self, thought_id: str) -> bool:
        """Remove a thought and all its connections"""
        if thought_id not in self.nodes:
            return False
            
        # Remove all edges involving this thought
        edges_to_remove = []
        for edge_key, edge in self.edges.items():
            if edge.source_id == thought_id or edge.target_id == thought_id:
                edges_to_remove.append(edge_key)
                
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
            
        # Remove from NetworkX if enabled
        if self.use_networkx and thought_id in self.nx_graph:
            self.nx_graph.remove_node(thought_id)
            
        # Remove the thought
        del self.nodes[thought_id]
        self.updated_at = time.time()
        return True
    
    def get_thought(self, thought_id: str) -> Optional[ThoughtNode]:
        """Get a thought by ID"""
        return self.nodes.get(thought_id)
    
    def get_parents(self, thought_id: str) -> List[ThoughtNode]:
        """Get all parent thoughts"""
        if thought_id not in self.nodes:
            return []
            
        parents = []
        for parent_id in self.nodes[thought_id].parents:
            if parent_id in self.nodes:
                parents.append(self.nodes[parent_id])
        return parents
    
    def get_children(self, thought_id: str) -> List[ThoughtNode]:
        """Get all child thoughts"""
        if thought_id not in self.nodes:
            return []
            
        children = []
        for child_id in self.nodes[thought_id].children:
            if child_id in self.nodes:
                children.append(self.nodes[child_id])
        return children
    
    def get_ancestors(self, thought_id: str, max_depth: int = 10) -> Set[str]:
        """Get all ancestor thoughts (recursive parents)"""
        if not self.use_networkx:
            # Fallback implementation
            ancestors = set()
            stack = [thought_id]
            depth = 0
            
            while stack and depth < max_depth:
                current = stack.pop()
                if current in self.nodes:
                    for parent_id in self.nodes[current].parents:
                        if parent_id not in ancestors:
                            ancestors.add(parent_id)
                            stack.append(parent_id)
                depth += 1
                
            return ancestors
        else:
            # Use NetworkX for efficient computation
            try:
                return set(nx.ancestors(self.nx_graph, thought_id))
            except nx.NetworkXError:
                return set()
    
    def get_descendants(self, thought_id: str, max_depth: int = 10) -> Set[str]:
        """Get all descendant thoughts (recursive children)"""
        if not self.use_networkx:
            # Fallback implementation
            descendants = set()
            stack = [thought_id]
            depth = 0
            
            while stack and depth < max_depth:
                current = stack.pop()
                if current in self.nodes:
                    for child_id in self.nodes[current].children:
                        if child_id not in descendants:
                            descendants.add(child_id)
                            stack.append(child_id)
                depth += 1
                
            return descendants
        else:
            # Use NetworkX for efficient computation
            try:
                return set(nx.descendants(self.nx_graph, thought_id))
            except nx.NetworkXError:
                return set()
    
    def find_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Find all paths between two thoughts"""
        if not self.use_networkx:
            # Simple path finding fallback
            return []
            
        try:
            paths = list(nx.all_simple_paths(self.nx_graph, source_id, target_id))
            return paths
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []
    
    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Get shortest path between two thoughts"""
        if not self.use_networkx:
            return None
            
        try:
            return nx.shortest_path(self.nx_graph, source_id, target_id)
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return None
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the thought graph"""
        if not self.use_networkx:
            return []
            
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
            return cycles
        except nx.NetworkXError:
            return []
    
    def get_root_thoughts(self) -> List[ThoughtNode]:
        """Get thoughts with no parents (root nodes)"""
        roots = []
        for thought in self.nodes.values():
            if len(thought.parents) == 0:
                roots.append(thought)
        return roots
    
    def get_leaf_thoughts(self) -> List[ThoughtNode]:
        """Get thoughts with no children (leaf nodes)"""
        leaves = []
        for thought in self.nodes.values():
            if len(thought.children) == 0:
                leaves.append(thought)
        return leaves
    
    def get_thoughts_by_type(self, thought_type: ThoughtType) -> List[ThoughtNode]:
        """Get all thoughts of a specific type"""
        return [thought for thought in self.nodes.values() 
                if thought.thought_type == thought_type]
    
    def get_high_confidence_thoughts(self, threshold: float = 0.7) -> List[ThoughtNode]:
        """Get thoughts with confidence above threshold"""
        return [thought for thought in self.nodes.values() 
                if thought.confidence >= threshold]
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate various graph metrics"""
        metrics = {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'density': 0.0,
            'avg_confidence': 0.0,
            'avg_relevance': 0.0,
            'has_cycles': False
        }
        
        if len(self.nodes) > 0:
            # Calculate average metrics
            total_confidence = sum(node.confidence for node in self.nodes.values())
            total_relevance = sum(node.relevance for node in self.nodes.values())
            
            metrics['avg_confidence'] = total_confidence / len(self.nodes)
            metrics['avg_relevance'] = total_relevance / len(self.nodes)
            
            # Calculate density
            max_edges = len(self.nodes) * (len(self.nodes) - 1)
            if max_edges > 0:
                metrics['density'] = len(self.edges) / max_edges
                
        # Check for cycles
        cycles = self.detect_cycles()
        metrics['has_cycles'] = len(cycles) > 0
        metrics['num_cycles'] = len(cycles)
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metrics': self.calculate_graph_metrics()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load graph from dictionary"""
        self.metadata = data.get('metadata', {})
        self.created_at = data.get('created_at', time.time())
        self.updated_at = data.get('updated_at', time.time())
        
        # Load nodes
        for node_data in data.get('nodes', {}).values():
            thought = ThoughtNode.from_dict(node_data)
            self.add_thought(thought)
            
        # Load edges
        for edge_data in data.get('edges', []):
            edge = ThoughtEdge(**edge_data)
            self.add_edge(edge)


class GraphOfThoughts:
    """
    Main Graph of Thoughts reasoning engine.
    
    This class orchestrates the creation, manipulation, and traversal
    of thought graphs to solve complex reasoning problems.
    """
    
    def __init__(self, max_thoughts: int = 1000, max_depth: int = 20):
        """
        Initialize the Graph of Thoughts engine.
        
        Args:
            max_thoughts: Maximum number of thoughts to generate
            max_depth: Maximum reasoning depth
        """
        self.graph = ThoughtGraph()
        self.max_thoughts = max_thoughts
        self.max_depth = max_depth
        
        # Reasoning state
        self.current_goal: Optional[str] = None
        self.goal_thoughts: List[str] = []
        self.reasoning_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.thought_quality_threshold = 0.5
        self.path_exploration_limit = 100
        
    def set_goal(self, goal_content: str, goal_context: Dict[str, Any] = None) -> str:
        """Set the reasoning goal and create a goal thought"""
        goal_thought = ThoughtNode(
            content=goal_content,
            thought_type=ThoughtType.GOAL,
            context=goal_context or {},
            relevance=1.0,
            confidence=1.0,
            creator="user"
        )
        
        self.graph.add_thought(goal_thought)
        self.current_goal = goal_thought.id
        self.goal_thoughts = [goal_thought.id]
        
        logger.info(f"Set reasoning goal: {goal_content}")
        return goal_thought.id
    
    def add_initial_observations(self, observations: List[str]) -> List[str]:
        """Add initial observations to start reasoning"""
        observation_ids = []
        
        for obs_content in observations:
            obs_thought = ThoughtNode(
                content=obs_content,
                thought_type=ThoughtType.OBSERVATION,
                confidence=0.8,  # High confidence in observations
                relevance=0.7,   # Moderate relevance initially
                creator="user"
            )
            
            self.graph.add_thought(obs_thought)
            observation_ids.append(obs_thought.id)
            
        logger.info(f"Added {len(observation_ids)} initial observations")
        return observation_ids
    
    def generate_thoughts(self, 
                         source_thoughts: List[str],
                         thought_generators: List[Callable],
                         max_new_thoughts: int = 10) -> List[str]:
        """
        Generate new thoughts from existing ones using thought generators.
        
        Args:
            source_thoughts: IDs of thoughts to generate from
            thought_generators: List of generator functions
            max_new_thoughts: Maximum new thoughts to generate
            
        Returns:
            List of new thought IDs
        """
        new_thought_ids = []
        
        for generator in thought_generators:
            if len(new_thought_ids) >= max_new_thoughts:
                break
                
            for source_id in source_thoughts:
                if len(new_thought_ids) >= max_new_thoughts:
                    break
                    
                source_thought = self.graph.get_thought(source_id)
                if not source_thought:
                    continue
                    
                try:
                    # Call the thought generator
                    generated_thoughts = generator(source_thought, self.graph)
                    
                    for new_thought in generated_thoughts:
                        if len(new_thought_ids) >= max_new_thoughts:
                            break
                            
                        # Add the new thought
                        if self.graph.add_thought(new_thought):
                            new_thought_ids.append(new_thought.id)
                            
                            # Create edge from source to new thought
                            edge = ThoughtEdge(
                                source_id=source_id,
                                target_id=new_thought.id,
                                edge_type="generates",
                                weight=new_thought.confidence
                            )
                            self.graph.add_edge(edge)
                            
                except Exception as e:
                    logger.error(f"Error in thought generator: {e}")
                    continue
                    
        logger.info(f"Generated {len(new_thought_ids)} new thoughts")
        return new_thought_ids
    
    def search_solution_paths(self, 
                            strategy: SearchStrategy = SearchStrategy.BEST_FIRST,
                            max_paths: int = 5) -> List[List[str]]:
        """
        Search for paths from observations to goals.
        
        Args:
            strategy: Search strategy to use
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths (each path is a list of thought IDs)
        """
        if not self.current_goal:
            logger.warning("No goal set for path search")
            return []
            
        # Get all observation thoughts as potential starting points
        observations = self.graph.get_thoughts_by_type(ThoughtType.OBSERVATION)
        if not observations:
            logger.warning("No observation thoughts found")
            return []
            
        paths = []
        
        for obs in observations:
            if len(paths) >= max_paths:
                break
                
            # Find paths from this observation to the goal
            obs_paths = self.graph.find_paths(obs.id, self.current_goal)
            
            # Apply search strategy to select best paths
            if strategy == SearchStrategy.BEST_FIRST:
                # Sort by path quality (average thought confidence)
                scored_paths = []
                for path in obs_paths:
                    score = self._calculate_path_score(path)
                    scored_paths.append((score, path))
                    
                scored_paths.sort(reverse=True)  # Highest score first
                paths.extend([path for score, path in scored_paths[:max_paths - len(paths)]])
                
            elif strategy == SearchStrategy.BREADTH_FIRST:
                # Take shorter paths first
                obs_paths.sort(key=len)
                paths.extend(obs_paths[:max_paths - len(paths)])
                
            else:
                # Default: just take first paths found
                paths.extend(obs_paths[:max_paths - len(paths)])
                
        return paths[:max_paths]
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """Calculate quality score for a reasoning path"""
        if not path:
            return 0.0
            
        total_score = 0.0
        valid_thoughts = 0
        
        for thought_id in path:
            thought = self.graph.get_thought(thought_id)
            if thought:
                # Weight different factors
                thought_score = (
                    thought.confidence * 0.4 +
                    thought.relevance * 0.3 +
                    thought.evidence_strength * 0.2 +
                    thought.novelty * 0.1
                )
                total_score += thought_score
                valid_thoughts += 1
                
        return total_score / max(valid_thoughts, 1)
    
    def evaluate_solutions(self, paths: List[List[str]]) -> List[Dict[str, Any]]:
        """Evaluate solution paths and rank them"""
        evaluations = []
        
        for i, path in enumerate(paths):
            path_score = self._calculate_path_score(path)
            
            # Additional path metrics
            path_length = len(path)
            avg_confidence = 0.0
            min_confidence = 1.0
            thought_types = set()
            
            for thought_id in path:
                thought = self.graph.get_thought(thought_id)
                if thought:
                    avg_confidence += thought.confidence
                    min_confidence = min(min_confidence, thought.confidence)
                    thought_types.add(thought.thought_type.value)
                    
            avg_confidence /= max(path_length, 1)
            
            evaluation = {
                'path_index': i,
                'path': path,
                'overall_score': path_score,
                'length': path_length,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'reasoning_diversity': len(thought_types),
                'completeness': self._assess_path_completeness(path)
            }
            
            evaluations.append(evaluation)
            
        # Sort by overall score
        evaluations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return evaluations
    
    def _assess_path_completeness(self, path: List[str]) -> float:
        """Assess how complete a reasoning path is"""
        # Check if path has diverse reasoning types
        thought_types = set()
        for thought_id in path:
            thought = self.graph.get_thought(thought_id)
            if thought:
                thought_types.add(thought.thought_type)
                
        # Desired reasoning progression
        desired_types = {
            ThoughtType.OBSERVATION,
            ThoughtType.HYPOTHESIS, 
            ThoughtType.DEDUCTION,
            ThoughtType.EVALUATION
        }
        
        overlap = len(thought_types.intersection(desired_types))
        return overlap / len(desired_types)
    
    def explain_reasoning(self, path: List[str]) -> str:
        """Generate human-readable explanation of a reasoning path"""
        if not path:
            return "No reasoning path provided."
            
        explanation = []
        explanation.append(f"Reasoning Path ({len(path)} steps):\n")
        
        for i, thought_id in enumerate(path, 1):
            thought = self.graph.get_thought(thought_id)
            if thought:
                confidence_indicator = "🟢" if thought.confidence > 0.7 else "🟡" if thought.confidence > 0.4 else "🔴"
                
                explanation.append(
                    f"{i}. [{thought.thought_type.value.upper()}] {confidence_indicator} "
                    f"{thought.content} (confidence: {thought.confidence:.2f})"
                )
                
                if thought.evidence:
                    explanation.append(f"   Evidence: {', '.join(thought.evidence[:2])}")
                    
        return "\n".join(explanation)
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the reasoning process"""
        stats = self.graph.calculate_graph_metrics()
        
        # Add GoT-specific statistics
        thought_type_counts = {}
        for thought_type in ThoughtType:
            thoughts = self.graph.get_thoughts_by_type(thought_type)
            thought_type_counts[thought_type.value] = len(thoughts)
            
        stats['thought_type_distribution'] = thought_type_counts
        stats['reasoning_goal'] = self.current_goal
        stats['total_reasoning_sessions'] = len(self.reasoning_history)
        
        if self.current_goal:
            goal_thought = self.graph.get_thought(self.current_goal)
            if goal_thought:
                stats['goal_content'] = goal_thought.content
                
        return stats
    
    def reset_reasoning(self):
        """Reset the reasoning state for a new problem"""
        self.graph = ThoughtGraph()
        self.current_goal = None
        self.goal_thoughts = []
        
        # Keep reasoning history for analysis
        if self.current_goal:
            session_record = {
                'goal': self.current_goal,
                'completed_at': time.time(),
                'final_stats': self.get_reasoning_statistics()
            }
            self.reasoning_history.append(session_record)
            
        logger.info("Reset reasoning state for new problem")
    
    def save_state(self, filepath: str):
        """Save the current reasoning state to file"""
        state_data = {
            'graph': self.graph.to_dict(),
            'current_goal': self.current_goal,
            'goal_thoughts': self.goal_thoughts,
            'reasoning_history': self.reasoning_history,
            'config': {
                'max_thoughts': self.max_thoughts,
                'max_depth': self.max_depth,
                'thought_quality_threshold': self.thought_quality_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
            
        logger.info(f"Saved reasoning state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load reasoning state from file"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
            
        # Load graph
        self.graph.from_dict(state_data.get('graph', {}))
        
        # Load reasoning state
        self.current_goal = state_data.get('current_goal')
        self.goal_thoughts = state_data.get('goal_thoughts', [])
        self.reasoning_history = state_data.get('reasoning_history', [])
        
        # Load configuration
        config = state_data.get('config', {})
        self.max_thoughts = config.get('max_thoughts', 1000)
        self.max_depth = config.get('max_depth', 20)
        self.thought_quality_threshold = config.get('thought_quality_threshold', 0.5)
        
        logger.info(f"Loaded reasoning state from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create a simple example
    got = GraphOfThoughts()
    
    # Set a glycoinformatics goal
    goal_id = got.set_goal(
        "Identify the glycan structure from mass spectrum data",
        {"spectrum_type": "MS/MS", "precursor_mass": 1234.56}
    )
    
    # Add initial observations
    observations = got.add_initial_observations([
        "Precursor ion at m/z 1234.56",
        "Fragment ions at m/z 666.33, 504.22, 342.11",
        "High intensity fragment at m/z 204.09 (HexNAc signature)"
    ])
    
    print(f"Created reasoning graph with {len(got.graph.nodes)} thoughts")
    print(f"Goal: {got.current_goal}")
    print(f"Observations: {observations}")
    
    # Get basic statistics
    stats = got.get_reasoning_statistics()
    print(f"Graph statistics: {stats}")