"""Processing pipeline coordinator for facial recognition system."""

import threading
import queue
import time
import logging
from typing import Dict, Any, List, Optional, Callable

class Pipeline:
    """Pipeline coordinator for multi-stage processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.num_workers = config.get('num_workers', 4)
        self.queue_size = config.get('queue_size', 10)
        self.logger = logging.getLogger(__name__)
        
        # Initialize queues
        self.input_queue = queue.Queue(maxsize=self.queue_size)
        self.output_queue = queue.Queue(maxsize=self.queue_size)
        
        # Initialize stages
        self.stages = []
        self.stage_names = []
        
        # Initialize threads
        self.threads = []
        self.running = False
        
    def add_stage(self, name: str, processor: Callable, 
                max_batch_size: int = 1) -> None:
        """Add processing stage to pipeline.
        
        Args:
            name: Stage name
            processor: Processing function
            max_batch_size: Maximum batch size for processing
        """
        self.stages.append({
            'name': name,
            'processor': processor,
            'max_batch_size': max_batch_size
        })
        self.stage_names.append(name)
        self.logger.info(f"Added pipeline stage: {name}")
        
    def start(self) -> None:
        """Start pipeline processing."""
        if not self.stages:
            self.logger.error("Cannot start pipeline with no stages")
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(len(self.stages)):
            stage = self.stages[i]
            for _ in range(self.num_workers):
                thread = threading.Thread(
                    target=self._stage_worker,
                    args=(i, stage),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
                
        self.logger.info(f"Pipeline started with {len(self.stages)} stages and {self.num_workers} workers per stage")
        
    def stop(self) -> None:
        """Stop pipeline processing."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
            
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Pipeline stopped")
        
    def _stage_worker(self, stage_idx: int, stage: Dict[str, Any]) -> None:
        """Worker function for pipeline stage.
        
        Args:
            stage_idx: Stage index
            stage: Stage configuration
        """
        name = stage['name']
        processor = stage['processor']
        max_batch_size = stage['max_batch_size']
        
        # Get input and output queues
        if stage_idx == 0:
            input_queue = self.input_queue
        else:
            input_queue = self.stages[stage_idx - 1]['output_queue']
            
        if stage_idx == len(self.stages) - 1:
            output_queue = self.output_queue
        else:
            if 'output_queue' not in stage:
                stage['output_queue'] = queue.Queue(maxsize=self.queue_size)
            output_queue = stage['output_queue']
            
        # Process items
        while self.running:
            try:
                # Get batch of items
                batch = []
                item = input_queue.get(timeout=0.1)
                batch.append(item)
                
                # Try to get more items for batch
                if max_batch_size > 1:
                    while len(batch) < max_batch_size:
                        try:
                            item = input_queue.get_nowait()
                            batch.append(item)
                        except queue.Empty:
                            break
                
                # Process batch
                start_time = time.time()
                try:
                    if len(batch) == 1:
                        result = processor(batch[0])
                        output_queue.put(result)
                    else:
                        results = processor(batch)
                        for result in results:
                            output_queue.put(result)
                except Exception as e:
                    self.logger.error(f"Error in stage {name}: {e}")
                    
                # Mark tasks as done
                for _ in range(len(batch)):
                    input_queue.task_done()
                    
                process_time = time.time() - start_time
                if process_time > 0.1:
                    self.logger.debug(f"Stage {name} processed {len(batch)} items in {process_time:.3f}s")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in stage {name} worker: {e}")
                
    def put(self, item: Any) -> bool:
        """Put item into pipeline.
        
        Args:
            item: Input item
            
        Returns:
            Success flag
        """
        try:
            self.input_queue.put(item, block=False)
            return True
        except queue.Full:
            return False
            
    def get(self, timeout: Optional[float] = 0.1) -> Optional[Any]:
        """Get item from pipeline.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Output item or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Pipeline statistics
        """
        stats = {
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'num_stages': len(self.stages),
            'num_workers': self.num_workers,
            'running': self.running
        }
        
        # Add stage queues
        for i, stage in enumerate(self.stages):
            if i < len(self.stages) - 1 and 'output_queue' in stage:
                stats[f'stage_{i}_queue_size'] = stage['output_queue'].qsize()
                
        return stats