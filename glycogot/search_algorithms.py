"""
Search Algorithms for Graph of Thoughts Implementation
=====================================================

This module implements various graph search algorithms specifically
designed for navigating and exploring thought graphs in reasoning tasks.
"""

import heapq
import random
import logging
import time
from typing import List, Dict, Set, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .graph_reasoning import ThoughtNode, ThoughtGraph, ThoughtType, SearchStrategy

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a graph search operation"""
    path: List[str]                        # Sequence of thought IDs
    cost: float                           # Total cost of the path
    nodes_explored: int                   # Number of nodes explored
    search_time: float                    # Time taken for search
    success: bool                         # Whether goal was reached
    metadata: Dict[str, Any]              # Additional search metadata


class SearchAlgorithm(ABC):
    """Abstract base class for thought graph search algorithms"""
    
    @abstractmethod
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """
        Search for paths from start nodes to nodes satisfying goal condition.
        
        Args:
            graph: The thought graph to search
            start_nodes: List of starting node IDs
            goal_condition: Function that returns True for goal nodes
            max_nodes: Maximum number of nodes to explore
            
        Returns:
            SearchResult containing the best path found
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of this search algorithm"""
        pass


class BreadthFirstSearch(SearchAlgorithm):
    """
    Breadth-First Search for thought graphs.
    
    Explores thoughts level by level, guaranteeing the shortest path
    in terms of number of reasoning steps.
    """
    
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform breadth-first search"""
        start_time = time.time()
        
        # Initialize search state
        queue = []
        visited = set()
        parent = {}
        nodes_explored = 0
        
        # Add all start nodes to queue
        for start_id in start_nodes:
            if start_id in graph.nodes:
                queue.append(start_id)
                visited.add(start_id)
                parent[start_id] = None
        
        # Search loop
        while queue and nodes_explored < max_nodes:
            current_id = queue.pop(0)  # FIFO for BFS
            current_thought = graph.get_thought(current_id)
            
            if not current_thought:
                continue
                
            nodes_explored += 1
            
            # Check if goal is reached
            if goal_condition(current_thought):
                path = self._reconstruct_path(parent, current_id)
                search_time = time.time() - start_time
                
                return SearchResult(
                    path=path,
                    cost=len(path),
                    nodes_explored=nodes_explored,
                    search_time=search_time,
                    success=True,
                    metadata={'algorithm': 'bfs', 'queue_size': len(queue)}
                )
            
            # Expand neighbors (children in thought graph)
            for child_id in current_thought.children:
                if child_id not in visited and child_id in graph.nodes:
                    visited.add(child_id)
                    parent[child_id] = current_id
                    queue.append(child_id)
        
        # No goal found
        search_time = time.time() - start_time
        return SearchResult(
            path=[],
            cost=float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=False,
            metadata={'algorithm': 'bfs', 'final_queue_size': len(queue)}
        )
    
    def _reconstruct_path(self, parent: Dict[str, str], goal_id: str) -> List[str]:
        """Reconstruct path from start to goal"""
        path = []
        current = goal_id
        
        while current is not None:
            path.append(current)
            current = parent.get(current)
            
        path.reverse()
        return path
    
    def get_algorithm_name(self) -> str:
        return "breadth_first_search"


class DepthFirstSearch(SearchAlgorithm):
    """
    Depth-First Search for thought graphs.
    
    Explores thoughts deeply before backtracking, good for finding
    any solution quickly but doesn't guarantee shortest path.
    """
    
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform depth-first search"""
        start_time = time.time()
        
        # Initialize search state
        stack = []
        visited = set()
        parent = {}
        nodes_explored = 0
        
        # Add all start nodes to stack
        for start_id in start_nodes:
            if start_id in graph.nodes:
                stack.append(start_id)
                visited.add(start_id)
                parent[start_id] = None
        
        # Search loop
        while stack and nodes_explored < max_nodes:
            current_id = stack.pop()  # LIFO for DFS
            current_thought = graph.get_thought(current_id)
            
            if not current_thought:
                continue
                
            nodes_explored += 1
            
            # Check if goal is reached
            if goal_condition(current_thought):
                path = self._reconstruct_path(parent, current_id)
                search_time = time.time() - start_time
                
                return SearchResult(
                    path=path,
                    cost=len(path),
                    nodes_explored=nodes_explored,
                    search_time=search_time,
                    success=True,
                    metadata={'algorithm': 'dfs', 'stack_size': len(stack)}
                )
            
            # Expand neighbors (children in thought graph)
            children = list(current_thought.children)
            children.reverse()  # Reverse to maintain consistent ordering
            
            for child_id in children:
                if child_id not in visited and child_id in graph.nodes:
                    visited.add(child_id)
                    parent[child_id] = current_id
                    stack.append(child_id)
        
        # No goal found
        search_time = time.time() - start_time
        return SearchResult(
            path=[],
            cost=float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=False,
            metadata={'algorithm': 'dfs', 'final_stack_size': len(stack)}
        )
    
    def _reconstruct_path(self, parent: Dict[str, str], goal_id: str) -> List[str]:
        """Reconstruct path from start to goal"""
        path = []
        current = goal_id
        
        while current is not None:
            path.append(current)
            current = parent.get(current)
            
        path.reverse()
        return path
    
    def get_algorithm_name(self) -> str:
        return "depth_first_search"


class BestFirstSearch(SearchAlgorithm):
    """
    Best-First Search using thought quality as heuristic.
    
    Prioritizes exploring high-quality thoughts first, using confidence
    and relevance scores to guide the search.
    """
    
    def __init__(self, heuristic_weights: Dict[str, float] = None):
        """
        Initialize best-first search.
        
        Args:
            heuristic_weights: Weights for combining thought attributes
        """
        self.weights = heuristic_weights or {
            'confidence': 0.4,
            'relevance': 0.4,
            'evidence_strength': 0.1,
            'novelty': 0.1
        }
    
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform best-first search"""
        start_time = time.time()
        
        # Initialize search state
        # Priority queue: (priority, node_id)
        priority_queue = []
        visited = set()
        parent = {}
        g_cost = {}  # Actual cost from start
        nodes_explored = 0
        
        # Add all start nodes to priority queue
        for start_id in start_nodes:
            if start_id in graph.nodes:
                start_thought = graph.get_thought(start_id)
                if start_thought:
                    priority = -self._calculate_heuristic(start_thought)  # Negative for max-heap behavior
                    heapq.heappush(priority_queue, (priority, start_id))
                    g_cost[start_id] = 0
                    parent[start_id] = None
        
        # Search loop
        while priority_queue and nodes_explored < max_nodes:
            current_priority, current_id = heapq.heappop(priority_queue)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_thought = graph.get_thought(current_id)
            
            if not current_thought:
                continue
                
            nodes_explored += 1
            
            # Check if goal is reached
            if goal_condition(current_thought):
                path = self._reconstruct_path(parent, current_id)
                search_time = time.time() - start_time
                
                return SearchResult(
                    path=path,
                    cost=g_cost.get(current_id, 0),
                    nodes_explored=nodes_explored,
                    search_time=search_time,
                    success=True,
                    metadata={
                        'algorithm': 'best_first', 
                        'final_priority': -current_priority,
                        'queue_size': len(priority_queue)
                    }
                )
            
            # Expand neighbors
            for child_id in current_thought.children:
                if child_id not in visited and child_id in graph.nodes:
                    child_thought = graph.get_thought(child_id)
                    if child_thought:
                        new_g_cost = g_cost[current_id] + 1
                        
                        if child_id not in g_cost or new_g_cost < g_cost[child_id]:
                            g_cost[child_id] = new_g_cost
                            parent[child_id] = current_id
                            
                            # Calculate priority using heuristic
                            priority = -self._calculate_heuristic(child_thought)
                            heapq.heappush(priority_queue, (priority, child_id))
        
        # No goal found
        search_time = time.time() - start_time
        return SearchResult(
            path=[],
            cost=float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=False,
            metadata={'algorithm': 'best_first', 'final_queue_size': len(priority_queue)}
        )
    
    def _calculate_heuristic(self, thought: ThoughtNode) -> float:
        """Calculate heuristic value for a thought"""
        return (
            thought.confidence * self.weights.get('confidence', 0.4) +
            thought.relevance * self.weights.get('relevance', 0.4) +
            thought.evidence_strength * self.weights.get('evidence_strength', 0.1) +
            thought.novelty * self.weights.get('novelty', 0.1)
        )
    
    def _reconstruct_path(self, parent: Dict[str, str], goal_id: str) -> List[str]:
        """Reconstruct path from start to goal"""
        path = []
        current = goal_id
        
        while current is not None:
            path.append(current)
            current = parent.get(current)
            
        path.reverse()
        return path
    
    def get_algorithm_name(self) -> str:
        return "best_first_search"


class AStarSearch(SearchAlgorithm):
    """
    A* Search for optimal pathfinding in thought graphs.
    
    Combines actual cost with heuristic estimate to find optimal paths
    efficiently.
    """
    
    def __init__(self, heuristic_function: Callable[[ThoughtNode, str], float] = None):
        """
        Initialize A* search.
        
        Args:
            heuristic_function: Function to estimate cost to goal
        """
        self.heuristic = heuristic_function or self._default_heuristic
    
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform A* search"""
        start_time = time.time()
        
        # Initialize search state
        open_set = []  # Priority queue: (f_score, node_id)
        closed_set = set()
        parent = {}
        g_score = {}  # Actual cost from start
        f_score = {}  # g_score + heuristic
        nodes_explored = 0
        
        # Add start nodes
        for start_id in start_nodes:
            if start_id in graph.nodes:
                g_score[start_id] = 0
                f_score[start_id] = self.heuristic(graph.get_thought(start_id), None)
                heapq.heappush(open_set, (f_score[start_id], start_id))
                parent[start_id] = None
        
        while open_set and nodes_explored < max_nodes:
            current_f, current_id = heapq.heappop(open_set)
            
            if current_id in closed_set:
                continue
                
            closed_set.add(current_id)
            current_thought = graph.get_thought(current_id)
            
            if not current_thought:
                continue
                
            nodes_explored += 1
            
            # Check if goal reached
            if goal_condition(current_thought):
                path = self._reconstruct_path(parent, current_id)
                search_time = time.time() - start_time
                
                return SearchResult(
                    path=path,
                    cost=g_score.get(current_id, 0),
                    nodes_explored=nodes_explored,
                    search_time=search_time,
                    success=True,
                    metadata={
                        'algorithm': 'a_star',
                        'final_f_score': current_f,
                        'open_set_size': len(open_set)
                    }
                )
            
            # Expand neighbors
            for child_id in current_thought.children:
                if child_id in closed_set or child_id not in graph.nodes:
                    continue
                    
                child_thought = graph.get_thought(child_id)
                if not child_thought:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current_id] + self._edge_cost(current_thought, child_thought)
                
                if child_id not in g_score or tentative_g < g_score[child_id]:
                    parent[child_id] = current_id
                    g_score[child_id] = tentative_g
                    f_score[child_id] = tentative_g + self.heuristic(child_thought, current_id)
                    
                    # Add to open set if not already there
                    heapq.heappush(open_set, (f_score[child_id], child_id))
        
        # No solution found
        search_time = time.time() - start_time
        return SearchResult(
            path=[],
            cost=float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=False,
            metadata={'algorithm': 'a_star', 'final_open_set_size': len(open_set)}
        )
    
    def _default_heuristic(self, thought: ThoughtNode, parent_id: Optional[str]) -> float:
        """Default heuristic based on thought quality"""
        # Higher quality thoughts are closer to goal
        quality = (thought.confidence + thought.relevance + thought.evidence_strength) / 3
        return 1.0 - quality  # Lower cost for higher quality
    
    def _edge_cost(self, from_thought: ThoughtNode, to_thought: ThoughtNode) -> float:
        """Calculate cost of moving between thoughts"""
        # Lower cost for high-confidence transitions
        confidence_factor = (from_thought.confidence + to_thought.confidence) / 2
        return 1.0 / max(confidence_factor, 0.1)
    
    def _reconstruct_path(self, parent: Dict[str, str], goal_id: str) -> List[str]:
        """Reconstruct path from start to goal"""
        path = []
        current = goal_id
        
        while current is not None:
            path.append(current)
            current = parent.get(current)
            
        path.reverse()
        return path
    
    def get_algorithm_name(self) -> str:
        return "a_star_search"


class MonteCarloTreeSearch(SearchAlgorithm):
    """
    Monte Carlo Tree Search for thought graphs.
    
    Uses random sampling and exploration to find good reasoning paths,
    particularly useful for complex or uncertain reasoning domains.
    """
    
    def __init__(self, exploration_weight: float = 1.4, max_iterations: int = 1000):
        """
        Initialize MCTS.
        
        Args:
            exploration_weight: UCB1 exploration parameter
            max_iterations: Maximum MCTS iterations
        """
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        
        # MCTS state tracking
        self.visit_count = {}
        self.total_reward = {}
        self.children = {}
        
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform Monte Carlo Tree Search"""
        start_time = time.time()
        
        # Initialize MCTS state
        self.visit_count = {}
        self.total_reward = {}
        self.children = {}
        
        # Pick best start node
        root_id = self._select_best_start_node(graph, start_nodes)
        if not root_id:
            return SearchResult([], float('inf'), 0, 0, False, {'algorithm': 'mcts'})
        
        nodes_explored = 0
        best_path = []
        best_score = 0
        
        # MCTS iterations
        for iteration in range(self.max_iterations):
            if nodes_explored >= max_nodes:
                break
                
            # Selection and expansion
            path = self._select_and_expand(graph, root_id, goal_condition)
            if not path:
                continue
                
            nodes_explored += len(path)
            
            # Simulation
            final_node_id = path[-1]
            final_thought = graph.get_thought(final_node_id)
            
            if final_thought and goal_condition(final_thought):
                # Found goal - this is our best path
                search_time = time.time() - start_time
                return SearchResult(
                    path=path,
                    cost=len(path),
                    nodes_explored=nodes_explored,
                    search_time=search_time,
                    success=True,
                    metadata={
                        'algorithm': 'mcts',
                        'iterations': iteration + 1,
                        'exploration_weight': self.exploration_weight
                    }
                )
            
            # Evaluate path quality
            path_score = self._evaluate_path(graph, path)
            
            # Backpropagation
            self._backpropagate(path, path_score)
            
            # Track best path found so far
            if path_score > best_score:
                best_score = path_score
                best_path = path.copy()
        
        # Return best path found
        search_time = time.time() - start_time
        return SearchResult(
            path=best_path,
            cost=len(best_path) if best_path else float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=len(best_path) > 0,
            metadata={
                'algorithm': 'mcts',
                'iterations': self.max_iterations,
                'best_score': best_score
            }
        )
    
    def _select_best_start_node(self, graph: ThoughtGraph, start_nodes: List[str]) -> Optional[str]:
        """Select the best starting node"""
        best_id = None
        best_score = -1
        
        for node_id in start_nodes:
            thought = graph.get_thought(node_id)
            if thought:
                score = thought.confidence + thought.relevance
                if score > best_score:
                    best_score = score
                    best_id = node_id
        
        return best_id
    
    def _select_and_expand(self, graph: ThoughtGraph, root_id: str, 
                          goal_condition: Callable[[ThoughtNode], bool]) -> List[str]:
        """Select path using UCB1 and expand if needed"""
        path = [root_id]
        current_id = root_id
        
        # Selection phase
        while current_id in self.children:
            if not self.children[current_id]:  # No children
                break
                
            # UCB1 selection
            current_id = self._ucb1_select(current_id)
            if current_id:
                path.append(current_id)
            else:
                break
        
        # Expansion phase
        current_thought = graph.get_thought(current_id)
        if current_thought and current_id not in self.children:
            # Initialize children
            self.children[current_id] = list(current_thought.children)
            
            # Pick random child for expansion
            if self.children[current_id]:
                child_id = random.choice(self.children[current_id])
                if child_id in graph.nodes:
                    path.append(child_id)
        
        return path
    
    def _ucb1_select(self, parent_id: str) -> Optional[str]:
        """Select child using UCB1 formula"""
        if parent_id not in self.children or not self.children[parent_id]:
            return None
        
        parent_visits = self.visit_count.get(parent_id, 0)
        if parent_visits == 0:
            return random.choice(self.children[parent_id])
        
        best_child = None
        best_value = -float('inf')
        
        for child_id in self.children[parent_id]:
            child_visits = self.visit_count.get(child_id, 0)
            
            if child_visits == 0:
                # Unvisited nodes have infinite UCB1 value
                return child_id
            
            child_reward = self.total_reward.get(child_id, 0) / child_visits
            exploration = self.exploration_weight * (
                (2 * math.log(parent_visits) / child_visits) ** 0.5
            )
            
            ucb1_value = child_reward + exploration
            
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_child = child_id
        
        return best_child
    
    def _evaluate_path(self, graph: ThoughtGraph, path: List[str]) -> float:
        """Evaluate quality of a reasoning path"""
        if not path:
            return 0.0
        
        total_score = 0.0
        valid_nodes = 0
        
        for node_id in path:
            thought = graph.get_thought(node_id)
            if thought:
                node_score = (
                    thought.confidence * 0.4 +
                    thought.relevance * 0.4 +
                    thought.evidence_strength * 0.2
                )
                total_score += node_score
                valid_nodes += 1
        
        return total_score / max(valid_nodes, 1)
    
    def _backpropagate(self, path: List[str], reward: float):
        """Backpropagate reward through the path"""
        for node_id in path:
            self.visit_count[node_id] = self.visit_count.get(node_id, 0) + 1
            self.total_reward[node_id] = self.total_reward.get(node_id, 0) + reward
    
    def get_algorithm_name(self) -> str:
        return "monte_carlo_tree_search"


class RandomWalkSearch(SearchAlgorithm):
    """
    Random walk search for exploration and baseline comparison.
    
    Performs random walks through the thought graph, useful for
    baseline comparison and discovering unexpected reasoning paths.
    """
    
    def __init__(self, num_walks: int = 100, max_walk_length: int = 20):
        """
        Initialize random walk search.
        
        Args:
            num_walks: Number of random walks to perform
            max_walk_length: Maximum length of each walk
        """
        self.num_walks = num_walks
        self.max_walk_length = max_walk_length
    
    def search(self, graph: ThoughtGraph, 
              start_nodes: List[str], 
              goal_condition: Callable[[ThoughtNode], bool],
              max_nodes: int = 1000) -> SearchResult:
        """Perform random walk search"""
        start_time = time.time()
        
        best_path = []
        best_score = 0
        nodes_explored = 0
        successful_walks = 0
        
        for walk in range(self.num_walks):
            if nodes_explored >= max_nodes:
                break
            
            # Pick random start node
            start_id = random.choice(start_nodes) if start_nodes else None
            if not start_id or start_id not in graph.nodes:
                continue
            
            # Perform random walk
            path = self._random_walk(graph, start_id, goal_condition)
            nodes_explored += len(path)
            
            # Check if goal reached
            if path:
                final_thought = graph.get_thought(path[-1])
                if final_thought and goal_condition(final_thought):
                    search_time = time.time() - start_time
                    return SearchResult(
                        path=path,
                        cost=len(path),
                        nodes_explored=nodes_explored,
                        search_time=search_time,
                        success=True,
                        metadata={
                            'algorithm': 'random_walk',
                            'successful_walk': walk + 1,
                            'total_walks': self.num_walks
                        }
                    )
                
                # Track best path by quality
                path_score = self._evaluate_walk(graph, path)
                if path_score > best_score:
                    best_score = path_score
                    best_path = path.copy()
                    successful_walks += 1
        
        # Return best path found
        search_time = time.time() - start_time
        return SearchResult(
            path=best_path,
            cost=len(best_path) if best_path else float('inf'),
            nodes_explored=nodes_explored,
            search_time=search_time,
            success=len(best_path) > 0,
            metadata={
                'algorithm': 'random_walk',
                'total_walks': self.num_walks,
                'successful_walks': successful_walks,
                'best_score': best_score
            }
        )
    
    def _random_walk(self, graph: ThoughtGraph, start_id: str, 
                    goal_condition: Callable[[ThoughtNode], bool]) -> List[str]:
        """Perform a single random walk"""
        path = [start_id]
        current_id = start_id
        
        for step in range(self.max_walk_length):
            current_thought = graph.get_thought(current_id)
            if not current_thought:
                break
            
            # Check if goal reached
            if goal_condition(current_thought):
                break
            
            # Pick random child
            children = list(current_thought.children)
            if not children:
                break
            
            next_id = random.choice(children)
            if next_id not in graph.nodes:
                break
                
            path.append(next_id)
            current_id = next_id
        
        return path
    
    def _evaluate_walk(self, graph: ThoughtGraph, path: List[str]) -> float:
        """Evaluate quality of a random walk"""
        if not path:
            return 0.0
        
        total_quality = 0.0
        for node_id in path:
            thought = graph.get_thought(node_id)
            if thought:
                total_quality += thought.confidence + thought.relevance
        
        return total_quality / len(path)
    
    def get_algorithm_name(self) -> str:
        return "random_walk_search"


# Search Algorithm Factory
class SearchAlgorithmFactory:
    """Factory for creating search algorithms"""
    
    @staticmethod
    def create_algorithm(strategy: SearchStrategy, **kwargs) -> SearchAlgorithm:
        """Create a search algorithm based on strategy"""
        if strategy == SearchStrategy.BREADTH_FIRST:
            return BreadthFirstSearch()
        elif strategy == SearchStrategy.DEPTH_FIRST:
            return DepthFirstSearch()
        elif strategy == SearchStrategy.BEST_FIRST:
            return BestFirstSearch(**kwargs)
        elif strategy == SearchStrategy.A_STAR:
            return AStarSearch(**kwargs)
        elif strategy == SearchStrategy.MONTE_CARLO:
            return MonteCarloTreeSearch(**kwargs)
        else:
            return BreadthFirstSearch()  # Default fallback
    
    @staticmethod
    def get_all_algorithms() -> List[SearchAlgorithm]:
        """Get instances of all available search algorithms"""
        return [
            BreadthFirstSearch(),
            DepthFirstSearch(),
            BestFirstSearch(),
            AStarSearch(),
            MonteCarloTreeSearch(),
            RandomWalkSearch()
        ]


# Utility functions for search operations
def compare_search_algorithms(graph: ThoughtGraph,
                            start_nodes: List[str],
                            goal_condition: Callable[[ThoughtNode], bool],
                            algorithms: List[SearchAlgorithm] = None) -> Dict[str, SearchResult]:
    """
    Compare multiple search algorithms on the same problem.
    
    Args:
        graph: The thought graph to search
        start_nodes: Starting node IDs
        goal_condition: Goal condition function
        algorithms: List of algorithms to compare (default: all)
        
    Returns:
        Dictionary mapping algorithm names to search results
    """
    if algorithms is None:
        algorithms = SearchAlgorithmFactory.get_all_algorithms()
    
    results = {}
    
    for algorithm in algorithms:
        try:
            result = algorithm.search(graph, start_nodes, goal_condition)
            results[algorithm.get_algorithm_name()] = result
        except Exception as e:
            logger.error(f"Error in {algorithm.get_algorithm_name()}: {e}")
            results[algorithm.get_algorithm_name()] = SearchResult(
                path=[], cost=float('inf'), nodes_explored=0, 
                search_time=0, success=False, 
                metadata={'error': str(e)}
            )
    
    return results


def create_goal_condition_from_content(target_content: str) -> Callable[[ThoughtNode], bool]:
    """Create a goal condition function based on content matching"""
    def goal_condition(thought: ThoughtNode) -> bool:
        return target_content.lower() in thought.content.lower()
    return goal_condition


def create_goal_condition_from_type(target_type: ThoughtType) -> Callable[[ThoughtNode], bool]:
    """Create a goal condition function based on thought type"""
    def goal_condition(thought: ThoughtNode) -> bool:
        return thought.thought_type == target_type
    return goal_condition


def create_goal_condition_high_confidence(threshold: float = 0.8) -> Callable[[ThoughtNode], bool]:
    """Create a goal condition for high-confidence thoughts"""
    def goal_condition(thought: ThoughtNode) -> bool:
        return thought.confidence >= threshold
    return goal_condition


# Example usage and testing
if __name__ == "__main__":
    import math  # Need this for MCTS UCB1
    from .graph_reasoning import ThoughtGraph, ThoughtNode, ThoughtType
    
    # Create test graph
    graph = ThoughtGraph()
    
    # Add some test thoughts
    obs1 = ThoughtNode(content="Observation 1", thought_type=ThoughtType.OBSERVATION, confidence=0.8)
    hyp1 = ThoughtNode(content="Hypothesis 1", thought_type=ThoughtType.HYPOTHESIS, confidence=0.6)
    goal = ThoughtNode(content="Goal reached", thought_type=ThoughtType.GOAL, confidence=0.9)
    
    graph.add_thought(obs1)
    graph.add_thought(hyp1)
    graph.add_thought(goal)
    
    # Connect thoughts
    obs1.add_child(hyp1.id)
    hyp1.add_parent(obs1.id)
    hyp1.add_child(goal.id)
    goal.add_parent(hyp1.id)
    
    # Test search algorithms
    start_nodes = [obs1.id]
    goal_condition = create_goal_condition_from_content("Goal reached")
    
    algorithms = SearchAlgorithmFactory.get_all_algorithms()
    results = compare_search_algorithms(graph, start_nodes, goal_condition, algorithms)
    
    for algo_name, result in results.items():
        print(f"\n{algo_name}:")
        print(f"  Success: {result.success}")
        print(f"  Path length: {len(result.path)}")
        print(f"  Nodes explored: {result.nodes_explored}")
        print(f"  Search time: {result.search_time:.4f}s")