# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hash bucket processor for efficient data packing."""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from itertools import islice
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import bisect

class HashBucketProcessor:
    """Hash bucket processors are used to handle large data files and perform efficient boxing"""
    
    DTYPE_SAMPLE_INFO = np.dtype([
        ("w", np.uint16),       # Used to store the weights of the ViT part (which can be the number of pixels of the ViT part or the processing capacity of the ViT part)
        ("l", np.uint16),       # For storing the types of tokens in the llm part (several acres of tokens in the LLM input part)
        ("name", "U256")        # sample‘s name
    ])

    def __init__(self, file_path: Union[str, Path], logger: Optional[logging.Logger] = None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"The file does not exist: {file_path}")
            
        self.hash_buckets = defaultdict(lambda: np.array([], dtype=self.DTYPE_SAMPLE_INFO))
        self.total_lines = 0
        self.hb2_keys = []   # Which powers of 2 can be divided by
        self._logger = logger or self._setup_default_logger()

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Set the default logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def estimate_memory_usage(self) -> int:
        """Estimate the current memory usage of the hash bucket"""
        total_size = sys.getsizeof(self.hash_buckets)
        for key, arr in self.hash_buckets.items():
            total_size += sys.getsizeof(key) + arr.nbytes
        return total_size
    
    def _count_file_lines(self) -> int:
        """Calculate the total number of lines in the file using a more efficient method"""
        try:
            with self.file_path.open('rb') as f:
                return sum(1 for _ in f)
        except Exception as e:
            self._logger.warning(f"Quick count failed. Use the standard method: {e}")
            with self.file_path.open('r', encoding='utf-8') as f:
                return sum(1 for _ in f)

    def _parse_line(self, line: str) -> Optional[Tuple[int, int, str]]:
        """Parse single-row data and return (w, l, name) or None"""
        line = line.strip()
        if ':' not in line:
            return None
            
        try:
            name, key_str = line.split(':', 1)
            key = int(key_str)
            if 0 <= key <= 65535:
                return (0, key, name)
        except (ValueError, IndexError):
            pass
        return None
        
    def _update_buckets(self, parsed_data: List[Tuple[int, int, str]]) -> None:
        """Update the hash bucket"""
        data_array = np.array(parsed_data, dtype=self.DTYPE_SAMPLE_INFO)
        unique_l_values = np.unique(data_array['l'])

        for l_val in unique_l_values:
            mask = data_array['l'] == l_val
            chunk = data_array[mask]
            
            if l_val in self.hash_buckets:
                self.hash_buckets[l_val] = np.concatenate([self.hash_buckets[l_val], chunk])
            else:
                self.hash_buckets[l_val] = chunk
                
    def build_buckets(self, chunk_size: int = 100000) -> None:
        """Build a hash bucket"""
        self.total_lines = self._count_file_lines()
        self._logger.info(f"Start processing the file. Total number of lines: {self.total_lines}")
        
        with self.file_path.open('r', encoding='utf-8') as file:
            with tqdm(total=self.total_lines, unit='line', desc='Build a hash bucket') as pbar:
                while True:
                    lines = list(islice(file, chunk_size))
                    if not lines:
                        break

                    pbar.update(len(lines))
                    
                    # Parallel data parsing
                    parsed_data = []
                    for line in lines:
                        parsed = self._parse_line(line)
                        if parsed:
                            parsed_data.append(parsed)

                    if parsed_data:
                        self._update_buckets(parsed_data)                                

    @staticmethod
    def factors_of_two(a: int, C: int) -> List[Tuple[int, int]]:
        """Return all pairs of (b, n) that satisfy b * 2^n = a and b > C"""
        if a < 0 or C < 0:
            raise ValueError("a must be a positive integer and C must be a non-negative integer")
        res = []
        n = 0
        b = a
        while b > C:
            res.append((b, n))
            if b & 1:
                break
            b >>= 1
            n += 1
        return res

    def find_items(self, capacity: int) -> defaultdict[np.ndarray]:
        """Search for eligible items from the hash bucket"""

        if not self.hash_buckets:
            self._logger.warning("The hash bucket is empty. Please build the hash bucket first")
            return
            
        for key, value in self.hash_buckets.items():
            if not isinstance(value, np.ndarray) or value.dtype != self.DTYPE_SAMPLE_INFO:
                raise TypeError(f"The hash bucket data format is incorrect,key={key}")
            break
        self.hb2_keys=[]
        min_l_value = min(self.hash_buckets.keys())
        valid_b_values = [b for b, _ in self.factors_of_two(capacity, min_l_value - 1)]

        for b in valid_b_values:
            if b in self.hash_buckets:
                self.hb2_keys.append(b)

        self._logger.info(f" find {len(self.hb2_keys)} valid bucket keys ")

    def delete_by_index(self, result: defaultdict[np.ndarray], key: int, index: int) -> None:
        """Delete elements by index"""
        if key in result and 0 <= index < len(result[key]):
            result[key] = np.delete(result[key], index)

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Obtain statistical information"""
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        memory_gb = self.estimate_memory_usage() / (1024**3)
        
        return {
            "bucket_count": len(self.hash_buckets),
            "total_items": total_items,
            "memory_usage_gb": memory_gb,
            "hb2_keys_count": [len(self.hb2_keys),self.hb2_keys],
            "file_lines": self.total_lines
        }

    def __len__(self) -> int:
        """Return the total number of data items"""
        return sum(len(arr) for arr in self.hash_buckets.values())

    def __repr__(self) -> str:
        return f"HashBucketProcessor(buckets={len(self.hash_buckets)}, items={len(self)})"    
        
    def summary(self) -> None:
        """Print summary information"""
        stats = self.get_statistics()
        print(f"=== Hash bucket processing digest  ===")
        print(f"hash bucket count: {stats['bucket_count']}")
        print(f"total data items: {stats['total_items']}")
        print(f"Memory usage: {stats['memory_usage_gb']:.2f} GB")
        print(f"valid bucket key: {stats['hb2_keys_count']}")
        print(f"line count: {stats['file_lines']}")    

    def _cleanup_empty_keys(self, verbose: bool = False) -> int:
        """
        Clean up keys with zero elements in the hash bucket
        
        Args:
            verbose: Whether to print cleanup details
        
        Returns:
            int: Number of empty keys deleted
        """
        # 1. Collect empty keys that need to be deleted
        empty_keys = []
        for key in list(self.hash_buckets.keys()):
            if len(self.hash_buckets[key]) == 0:
                empty_keys.append(key)
        
        # 2. Delete empty keys
        for key in empty_keys:
            del self.hash_buckets[key]
        
        # 3. Log the operation
        if verbose or empty_keys:
            self._logger.info(f"Cleaning empty keys: Deleted {len(empty_keys)} empty keys")
            if verbose and empty_keys:
                self._logger.debug(f"Deleted keys: {sorted(empty_keys)}")
        
        return len(empty_keys)   

    def update_hash_buckets(self, remove_empty: bool = True, verbose: bool = False) -> dict:
        """
        Update the hash bucket structure, including cleaning empty keys and collecting statistics
        
        Args:
            remove_empty: Whether to delete empty keys
            verbose: Whether to print detailed information
        
        Returns:
            dict: Updated statistics
        """
        # 1. Basic statistics
        stats = {
            'before': {
                'total_keys': len(self.hash_buckets),
                'total_items': sum(len(arr) for arr in self.hash_buckets.values()),
                'empty_keys': sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
            }
        }
        
        # 2. Optional: Delete empty keys
        removed_keys = 0
        if remove_empty:
            removed_keys = self._cleanup_empty_keys(verbose)
        
        # 3. Statistics after update
        stats['after'] = {
            'total_keys': len(self.hash_buckets),
            'total_items': sum(len(arr) for arr in self.hash_buckets.values()),
            'empty_keys': sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
        }
        
        # 4. Calculate changes
        stats['changes'] = {
            'keys_removed': removed_keys,
            'items_removed': stats['before']['total_items'] - stats['after']['total_items']
        }
        
        # 5. Log the operation
        if verbose or stats['changes']['keys_removed'] > 0:
            self._logger.info("Hash bucket update completed:")
            self._logger.info(f"  Number of keys: {stats['before']['total_keys']} → {stats['after']['total_keys']}")
            self._logger.info(f"  Total items: {stats['before']['total_items']} → {stats['after']['total_items']}")
            self._logger.info(f"  Empty keys removed: {stats['changes']['keys_removed']}")
        
        return stats

    def get_hash_buckets_summary(self) -> dict:
        """
        Get summary information of hash buckets
        
        Returns:
            dict: Dictionary containing detailed statistics
        """
        # Basic statistics
        total_keys = len(self.hash_buckets)
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        empty_keys = sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
        
        # Statistics by size category
        size_distribution = {
            'large': 0,    # >= 8192
            'medium': 0,   # 2048-8192
            'small': 0     # < 2048
        }
        
        items_by_size = {
            'large': 0,
            'medium': 0,
            'small': 0
        }
        
        for key, arr in self.hash_buckets.items():
            count = len(arr)
            if key >= 8192:
                size_distribution['large'] += 1
                items_by_size['large'] += count
            elif key >= 2048:
                size_distribution['medium'] += 1
                items_by_size['medium'] += count
            else:
                size_distribution['small'] += 1
                items_by_size['small'] += count
        
        # Return complete summary
        return {
            'basic': {
                'total_keys': total_keys,
                'total_items': total_items,
                'empty_keys': empty_keys,
                'non_empty_keys': total_keys - empty_keys
            },
            'size_distribution': size_distribution,
            'items_by_size': items_by_size,
            'memory_usage': self.estimate_memory_usage()
        }      
    
    def print_example(self, key: int) -> None:
        """Print example data"""
        if key in self.hash_buckets:
            arr = self.hash_buckets[key]
            print(f"Key {key} data count: {len(arr)}")
            print("First 3 data items:")
            for item in arr[:3]:
                print(f"  w: {item['w']}, l: {item['l']}, name: {item['name']}")
        else:
            print(f"Key {key} does not exist.")
  
    def pack_with_deletion(self, box_capacity: int = 16384) -> List[np.ndarray]:
        """Pack by capacity, prioritize diversity, and delete used elements from the original bucket immediately after packing
        (Used for separately handling keys where (box_capacity/key) == 2^n)
        When encountering a non-full box, change the packing strategy once
        """
        from collections import deque
    
        boxes = []
    
        # Maintain a deque for each key for easy popping (only consider buckets with existing elements)
        key_queues = {k: deque(enumerate(self.hash_buckets[k])) 
                      for k in self.hb2_keys 
                      if k in self.hash_buckets and len(self.hash_buckets[k]) > 0}
    
        while any(key_queues.values()):
            current_box_items = []
            current_sum = 0
            used_indices = defaultdict(list)  # key -> list of indices to delete
    
            keys_to_try = deque(sorted(key_queues.keys()))
    
            while keys_to_try and current_sum < box_capacity:
                key = keys_to_try.popleft()
                queue = key_queues[key]
                if not queue:
                    continue
    
                idx, item = queue[0]
                l_val = key #item['l']
                if current_sum + l_val <= box_capacity:
                    queue.popleft()
                    current_box_items.append(item)
                    current_sum += l_val
                    used_indices[key].append(idx)
    
                    # If the key has remaining elements, put it back at the end of the queue
                    if queue:
                        keys_to_try.append(key)
    
            if current_box_items and current_sum==box_capacity:
                # Full box: output and delete
                boxes.append(np.array(current_box_items, dtype=self.DTYPE_SAMPLE_INFO))
    
                # Delete used elements from self.hash_buckets
                for key, indices in used_indices.items():
                    indices = sorted(indices, reverse=True)
                    for idx in indices:
                        self.hash_buckets[key] = np.delete(self.hash_buckets[key], idx)
    
                    # Update the deque in key_queues
                    key_queues[key] = deque(enumerate(self.hash_buckets[key]))
            else:
                # Add a judgment: if the number of elements in each queue is exactly the same, change the packing strategy once
                self._logger.info(f"Current box not full: {current_sum}")
                self._logger.info(f"Current box items: {current_box_items}")
                
                left_elems = [len(self.hash_buckets[k]) for k in self.hb2_keys if k in self.hash_buckets and len(self.hash_buckets[k])>0]
                # Remaining keys for packing
                left_keys = [k for k in self.hb2_keys if k in self.hash_buckets and len(self.hash_buckets[k])>0]
                print(f"Remaining keys and their element counts: (keys, counts):({left_keys},{left_elems})")
                if len(set(left_elems)) == 1:
                    self._logger.info(f"Change packing strategy to break the cycle")
                    b_succeed=False
                    # todo ...... Packing without considering diversity
                    current_box2 = []
                    current_sum2 = 0
                    used_keys_num = defaultdict(int)   # Record how many elements are used from this bucket
                    for key2 in left_keys:   # Take one bucket
                        if b_succeed:   # Only pack one
                            print(f"Changed strategy packing succeeded:")
                            break
                        arr2 = self.hash_buckets[key2]
                        l_val2 = key2
                        for item2 in arr2:
                            if current_sum2 + l_val2 <= box_capacity:
                                current_box2.append(item2)
                                current_sum2 += l_val2
                                used_keys_num[key2] += 1
                                
                                if current_sum2==box_capacity:
                                    boxes.append(np.array(current_box2, dtype=self.DTYPE_SAMPLE_INFO))
                                    current_box2 = []
                                    current_sum2 = 0
                                    for kkey, knum in used_keys_num.items():
                                        for _ in range(knum):
                                            self.hash_buckets[kkey] = np.delete(self.hash_buckets[kkey], 0)
                                        key_queues[kkey] = deque(enumerate(self.hash_buckets[key]))
                                    
                                    print(f"Changed strategy packing succeeded:{boxes[-1]}")
                                    used_keys_num = defaultdict(int)
                                    b_succeed = True
                                    break
                            else:
                                current_box2 = []
                                current_sum2 = 0
                                used_keys_num = defaultdict(int)
                                b_succeed = False
                                print(f"Changed strategy packing failed!")
                                break
                    pass
                else:
                    print(f"num of left_elems:{left_elems}")
    
        return boxes

    def pack_with_deletion_recursion(self, box_capacity: int = 16384) -> List[np.ndarray]:
        """Recursive diversity-first packing: Only output/delete full boxes, all non-full boxes are mixed and repacked until only one non-full box remains.
        (Used for separately handling keys where (box_capacity/key)==2^n)
        Implemented recursively
        """
        from collections import deque, defaultdict
        def recursive_diversity_pack(key_queues):
            boxes = []
            not_full_items = []
            print("----------- pack_with_deletion_recursion -----------")
            while any(key_queues.values()):
                current_box = []
                current_sum = 0
                used_indices = defaultdict(list)
                keys_to_try = deque(sorted(key_queues.keys()))
    
                # Diversity-first: Take from different buckets each round
                while keys_to_try and current_sum < box_capacity:
                    key = keys_to_try.popleft()
                    queue = key_queues[key]
                    if not queue:
                        continue
                    idx, item = queue[0]
                    l_val = item['l']
                    if current_sum + l_val <= box_capacity:
                        queue.popleft()
                        current_box.append((key, idx, item))
                        current_sum += l_val
                        used_indices[key].append(idx)
                        if queue:
                            keys_to_try.append(key)
    
                if current_sum == box_capacity:
                    # Full box, output and record indices to delete
                    boxes.append(np.array([item for _, _, item in current_box], dtype=self.DTYPE_SAMPLE_INFO))
                    for key, indices in used_indices.items():
                        # Delete used elements
                        indices = sorted(indices, reverse=True)
                        for idx in indices:
                            self.hash_buckets[key] = np.delete(self.hash_buckets[key], idx)
                        # Update key_queues
                        key_queues[key] = deque(enumerate(self.hash_buckets[key]))
                elif current_box:
                    # Non-full box, temporarily store
                    not_full_items.extend(current_box)
    
            return boxes, not_full_items
    
        # Initialize key_queues
        key_queues = {k: deque(enumerate(self.hash_buckets[k])) for k in self.hb2_keys if k in self.hash_buckets}
        boxes, not_full_items = recursive_diversity_pack(key_queues)
    
        # Mix all non-full box elements for recursive packing
        while not_full_items:
            # Mix all remaining elements and re-bucket
            mixed = defaultdict(list)
            for _, _, item in not_full_items:
                mixed[item['l']].append(item)
            key_queues = {k: deque(enumerate(np.array(v, dtype=self.DTYPE_SAMPLE_INFO))) for k, v in mixed.items()}
            new_boxes, new_not_full_items = recursive_diversity_pack(key_queues)
            boxes.extend(new_boxes)
            if not new_boxes or not new_not_full_items:
                break
            not_full_items = new_not_full_items
        return boxes, not_full_items

    def pack_large_seed_parallel_multithread(self, box_capacity: int = 16384, min_ratio: float = 0.95, 
                                           max_workers: int = None) -> List[np.ndarray]:
        """
        Multithreaded version (processing elements after pack_with_deletion): Large seed parallel packing, small elements as shared resources, real-time element deletion
         (No restrictions on the number of items in a box, will be relatively faster)
        Parameters:
            box_capacity: Box capacity
            min_ratio: Minimum loading rate threshold
            max_workers: Maximum number of threads, automatically set to CPU core count when None
        
        Returns:
            List[np.ndarray]: List of successfully packed boxes
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 8)  # Limit maximum number of threads
        
        half = box_capacity // 2
        # half = 4096
        large_keys = [k for k in self.hash_buckets.keys() if k >= half]
        small_keys = [k for k in self.hash_buckets.keys() if k < half]
        
        if not large_keys:
            self._logger.warning("No large seed elements found")
            return []
    
        # 1. Thread-safe shared resource manager
        class SharedResourceManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()  # Reentrant lock
                self.hash_buckets = hash_buckets  # Direct reference to original hash buckets
                self.small_keys = small_keys
                self.large_keys = large_keys
                
                # Initialize available small keys
                self.available_small_keys = sorted([
                    k for k in small_keys 
                    if k in hash_buckets and len(hash_buckets[k]) > 0
                ])
                
                # Statistical information
                self.total_processed = 0
                self.successful_boxes = 0
                
            def get_seed_item(self, seed_key: int) -> tuple:
                """Thread-safely get seed element"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> tuple:
                """Thread-safely get and delete an element from specified key"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # If this key's bucket is empty, remove from available_small_keys
                        if (len(self.hash_buckets[target_key]) == 0 and 
                            target_key in self.available_small_keys):
                            self.available_small_keys.remove(target_key)
                        
                        return True, item
                    return False, None
            
            def get_available_small_keys(self) -> List[int]:
                """Get current available small key list"""
                with self.lock:
                    return self.available_small_keys.copy()
            
            def update_stats(self, success: bool):
                """Update statistical information"""
                with self.lock:
                    self.total_processed += 1
                    if success:
                        self.successful_boxes += 1
                    
            def get_stats(self) -> dict:
                """Get statistical information"""
                with self.lock:
                    small_items_count = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    large_items_count = sum(
                        len(self.hash_buckets[k]) for k in self.large_keys 
                        if k in self.hash_buckets
                    )
                    
                    return {
                        'small_items_remaining': small_items_count,
                        'large_items_remaining': large_items_count,
                        'available_small_keys': len(self.available_small_keys),
                        'total_processed': self.total_processed,
                        'successful_boxes': self.successful_boxes,
                        'success_rate': (self.successful_boxes / max(1, self.total_processed))
                    }
    
        # 2. Binary search function
        def search_for_fit_key(available_keys: List[int], remaining_capacity: int) -> int:
            """Binary search for the largest key that can fit in available keys"""
            if not available_keys:
                return -1
            index = bisect.bisect(available_keys, remaining_capacity)
            return -1 if index == 0 else (index - 1)
    
        # 3. Single seed packing function
        def pack_single_seed(seed_key: int, shared_manager: SharedResourceManager, 
                            thread_id: int) -> tuple:
            """Pack items for a single seed"""
            try:
                # Get seed item
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    return False, None, thread_id, 0, "No available seed"
                
                current_box = [seed_item]
                remaining_capacity = box_capacity - seed_key
                items_added = 1
            
                # Greedy packing: prioritize larger elements
                max_iterations = 1000  # Prevent infinite loops
                iteration = 0
                
                while remaining_capacity > 0 and iteration < max_iterations:
                    iteration += 1
                    
                    # Get currently available small keys
                    available_keys = shared_manager.get_available_small_keys()
                    if not available_keys:
                        break
                    
                    # Binary search for the largest fittable key
                    best_key_index = search_for_fit_key(available_keys, remaining_capacity)
                    if best_key_index == -1:
                        break
                    
                    best_key = available_keys[best_key_index]
                    
                    # Attempt to get element by key
                    success, item = shared_manager.get_item_by_key(best_key)
                    if not success:
                        continue  # Key already consumed by another thread, retry
                    
                    current_box.append(item)
                    remaining_capacity -= best_key
                    items_added += 1
                    
                    # Stop if fully packed
                    if remaining_capacity == 0:
                        break
                
                # Check loading ratio
                current_capacity = box_capacity - remaining_capacity
                is_successful = current_capacity >= min_ratio * box_capacity
                
                result_box = current_box if is_successful else None
                load_ratio = current_capacity / box_capacity
                
                return (is_successful, result_box, thread_id, current_capacity, 
                       f"Load ratio:{load_ratio:.1%}, Items count:{items_added}")
                
            except Exception as e:
                return False, None, thread_id, 0, f"Packing exception: {str(e)}"
    
        # 4. Prepare all large seed tasks
        seed_tasks = []
        total_large_items = 0
        
        for key in large_keys:
            if key in self.hash_buckets:
                count = len(self.hash_buckets[key])
                total_large_items += count
                # Create a packing task for each large element
                for _ in range(count):
                    seed_tasks.append(key)
    
        if not seed_tasks:
            self._logger.warning("No available large seed elements")
            return []
    
        # 5. Initialize shared resource manager
        shared_manager = SharedResourceManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_stats()
        
        self._logger.info(f"Starting multithreaded packing:")
        self._logger.info(f"  Large seed tasks: {len(seed_tasks)}")
        self._logger.info(f"  Thread count: {max_workers}")
        self._logger.info(f"  Target capacity: {box_capacity}")
        self._logger.info(f"  Minimum load ratio: {min_ratio:.1%}")
        self._logger.info(f"  Small elements: {initial_stats['small_items_remaining']}")
    
        # 6. Multithreaded Execution
        output_boxes = []
        failed_reasons = defaultdict(int)
        start_time = time.time()
    
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="PackWorker") as executor:
            
            # Submit all tasks
            future_to_task = {}
            for i, seed_key in enumerate(seed_tasks):
                future = executor.submit(pack_single_seed, seed_key, shared_manager, i)
                future_to_task[future] = (seed_key, i)
            
            # Process results
            with tqdm(total=len(seed_tasks), unit='seed', 
                     desc=f'Multithreaded Packing', dynamic_ncols=True) as pbar:
                
                for future in as_completed(future_to_task):
                    seed_key, task_id = future_to_task[future]
                    
                    try:
                        success, box, thread_id, capacity, info = future.result(timeout=30)
                        
                        shared_manager.update_stats(success)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        else:
                            failed_reasons[info] += 1
                        
                        pbar.update(1)
                        
                        # Update description every 50 tasks
                        if pbar.n % 50 == 0:
                            current_stats = shared_manager.get_stats()
                            pbar.set_description(
                                f'Packing progress(Successful:{current_stats["successful_boxes"]}, '
                                f'Success rate:{current_stats["success_rate"]:.1%}, '
                                f'Remaining small items:{current_stats["small_items_remaining"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"Task {task_id} (seed key={seed_key}) failed: {e}")
                        failed_reasons[f"Execution exception: {str(e)}"] += 1
                        pbar.update(1)
    
        end_time = time.time()
        
        # 7. Output detailed statistics
        final_stats = shared_manager.get_stats()
        
        if output_boxes:
            total_items = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items / len(output_boxes)
            total_capacity_used = len(output_boxes) * box_capacity
            
            self._logger.info(f"Multithreaded packing completed:")
            self._logger.info(f"  Total time: {end_time - start_time:.2f} seconds")
            self._logger.info(f"  Successful boxes: {len(output_boxes)}")
            self._logger.info(f"  Overall success rate: {final_stats['success_rate']:.2%}")
            self._logger.info(f"  Average items per box: {avg_items_per_box:.1f}")
            self._logger.info(f"  Total items used: {total_items}")
            self._logger.info(f"  Remaining small items: {final_stats['small_items_remaining']}")
            self._logger.info(f"  Remaining small keys: {final_stats['available_small_keys']}")
            
            if failed_reasons:
                self._logger.info(f"  Failure reason statistics:")
                for reason, count in failed_reasons.items():
                    self._logger.info(f"     {reason}: {count} times")
        else:
            self._logger.warning("No items were successfully packed")
            self._logger.info(f"Failure reasons: {dict(failed_reasons)}")
    
        return output_boxes

    def pack_with_min_items_constraint_multithread(self, box_capacity: int = 16384, 
                                                 min_items: int = 10, min_ratio: float = 0.95,
                                                 max_workers: int = None) -> List[np.ndarray]:
        """
        Multi-threaded multi-constraint bin packing: capacity constraint + minimum item count constraint
        (Adds a minimum item count limit for each bin to ensure subsequent attn times are as close as possible)
        Parameters:
            box_capacity: bin capacity
            min_items: minimum number of items per bin
            min_ratio: minimum loading ratio threshold
            max_workers: maximum number of threads
        
        Returns:
            List[np.ndarray]: list of bins satisfying all constraints
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 6)  # Constrained problem is computationally intensive, reduce thread count
        
        half = box_capacity // 2
        # half = 4096
        print(f"Seed screening parameter half:{half}")
        large_keys = [k for k in self.hash_buckets.keys() if k >= half]
        small_keys = [k for k in self.hash_buckets.keys() if k < half]
        
        if not large_keys:
            self._logger.warning("No large seed elements found")
            return []
    
        # 1. Seed potential evaluator
        class SeedPotentialAnalyzer:
            def __init__(self, hash_buckets, small_keys, box_capacity, min_items):
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.box_capacity = box_capacity
                self.min_items = min_items
                
            def calculate_potential(self, seed_key: int) -> float:
                """Calculate the packing success potential of a seed"""
                remaining_capacity = self.box_capacity - seed_key
                
                if remaining_capacity <= 0:
                    return 0.0
                
                # Count distribution of small elements
                available_small_items = sum(
                    len(self.hash_buckets[k]) for k in self.small_keys 
                    if k in self.hash_buckets
                )
                
                if available_small_items == 0:
                    return 0.0
                
                # Estimate number of items that can be packed (conservative estimate)
                min_small_key = min(self.small_keys) if self.small_keys else remaining_capacity
                max_possible_items = remaining_capacity // min_small_key
                
                # Consider practical availability (not all small keys have elements)
                practical_items = min(max_possible_items, available_small_items // 2)
                total_items = practical_items + 1  # +1 for seed
                
                # Potential score
                count_score = min(total_items / self.min_items, 1.0) if self.min_items > 0 else 1.0
                capacity_score = seed_key / self.box_capacity
                diversity_score = len([k for k in self.small_keys if k <= remaining_capacity]) / len(self.small_keys) if self.small_keys else 0
                
                return count_score * 0.5 + capacity_score * 0.3 + diversity_score * 0.2

        # 2. Enhanced shared resource manager
        class EnhancedSharedManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.large_keys = large_keys
                
                # Maintain statistics of available keys
                self.key_stats = {}
                self._update_key_stats()
                
                # Performance statistics
                self.stats = {
                    'total_attempts': 0,
                    'successful_boxes': 0,
                    'failed_by_count': 0,
                    'failed_by_ratio': 0,
                    'failed_by_capacity': 0
                }
            
            def _update_key_stats(self):
                """Update key statistics"""
                self.key_stats = {}
                for k in self.small_keys:
                    if k in self.hash_buckets and len(self.hash_buckets[k]) > 0:
                        self.key_stats[k] = len(self.hash_buckets[k])
                        
            def get_seed_item(self, seed_key: int) -> Tuple[bool, Optional[np.record]]:
                """Get seed element"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> Tuple[bool, Optional[np.record]]:
                """Get element by specified key"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # Update statistics
                        if target_key in self.key_stats:
                            self.key_stats[target_key] -= 1
                            if self.key_stats[target_key] <= 0:
                                del self.key_stats[target_key]
                        
                        return True, item
                    return False, None
            
            def get_available_keys_with_counts(self) -> Dict[int, int]:
                """Get available keys and their element counts"""
                with self.lock:
                    return self.key_stats.copy()
            
            def rollback_items(self, items_to_rollback: List[Tuple[int, np.record]]):
                """Rollback elements from failed packing"""
                with self.lock:
                    for key, item in reversed(items_to_rollback):  # Rollback in reverse order
                        self.hash_buckets[key] = np.insert(self.hash_buckets[key], 0, item)
                        # Update statistics
                        if key in self.small_keys:
                            self.key_stats[key] = self.key_stats.get(key, 0) + 1
            
            def update_stats(self, result_type: str):
                """Update statistics"""
                with self.lock:
                    self.stats['total_attempts'] += 1
                    if result_type in self.stats:
                        self.stats[result_type] += 1
            
            def get_current_stats(self) -> Dict:
                """Get current statistics"""
                with self.lock:
                    total_small_items = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    return {
                        **self.stats,
                        'remaining_small_items': total_small_items,
                        'available_key_types': len(self.key_stats)
                    }

        # 3. Intelligent packing strategy
        def is_feasible_quick_check(remaining_capacity: int, current_items: int, 
                                   available_keys: Dict[int, int], min_items: int) -> bool:
            """Quick feasibility check"""
            if current_items >= min_items:
                return True
                
            needed_items = min_items - current_items
            if not available_keys:
                return False
            
            # Greedy estimation: prioritize small keys
            sorted_keys = sorted(available_keys.keys())
            possible_items = 0
            remaining_cap = remaining_capacity
            
            for key in sorted_keys:
                if remaining_cap <= 0:
                    break
                max_from_this_key = min(remaining_cap // key, available_keys[key])
                possible_items += max_from_this_key
                remaining_cap -= max_from_this_key * key
                
                if possible_items >= needed_items:
                    return True
                    
            return False
    
        def select_optimal_key(strategy: str, available_keys: Dict[int, int], 
                              remaining_capacity: int, current_items: int, min_items: int) -> Optional[int]:
            """Select optimal key based on strategy"""
            suitable_keys = [k for k in available_keys.keys() 
                            if k <= remaining_capacity and available_keys[k] > 0]
            if not suitable_keys:
                return None
            
            if strategy == "prioritize_count":
                # Prioritize count: select smallest key
                return min(suitable_keys)
            elif strategy == "prioritize_capacity":
                # Prioritize capacity: select largest key
                return max(suitable_keys)
            elif strategy == "balanced":
                # Balanced strategy: select medium size, but consider available quantity
                suitable_keys.sort()
                # Prefer keys with more available elements
                key_scores = [(k, available_keys[k] * (remaining_capacity / k)) for k in suitable_keys]
                key_scores.sort(key=lambda x: x[1], reverse=True)
                return key_scores[0][0]
            else:
                return suitable_keys[0]

        # 4. Core packing function
        def pack_single_seed_with_constraints(seed_key: int, shared_manager: EnhancedSharedManager, 
                                            thread_id: int) -> Tuple:
            """Single seed packing with constraints"""
            try:
                # Get seed
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    shared_manager.update_stats('failed_by_capacity')
                    return False, None, thread_id, 0, 0, "No available seed"
                
                current_box = [seed_item]
                used_items = [(seed_key, seed_item)]  # For rollback
                remaining_capacity = box_capacity - seed_key
                items_count = 1
                
                max_iterations = min_items * 16  # Prevent infinite loop
                iteration = 0
                
                # Main packing loop
                while (remaining_capacity > 0 and 
                       items_count < min_items * 8 and  # Allow exceeding minimum count
                       iteration < max_iterations):
                    
                    iteration += 1
                    available_keys = shared_manager.get_available_keys_with_counts()
                    
                    # Quick feasibility check
                    if (items_count < min_items and 
                        not is_feasible_quick_check(remaining_capacity, items_count, 
                                                  available_keys, min_items)):
                        # Cannot reach minimum item count, exit early
                        shared_manager.rollback_items(used_items)
                        shared_manager.update_stats('failed_by_count')
                        return False, None, thread_id, 0, items_count, f"Cannot reach {min_items} items"
                    
                    # Dynamic strategy selection
                    if items_count < min_items * 0.8:
                        strategy = "prioritize_count"
                    elif items_count < min_items:
                        strategy = "balanced"
                    else:
                        strategy = "prioritize_capacity"
                    
                    # Select next key
                    target_key = select_optimal_key(strategy, available_keys, 
                                                  remaining_capacity, items_count, min_items)
                    if target_key is None:
                        break
                    
                    # Get element
                    success, item = shared_manager.get_item_by_key(target_key)
                    if not success:
                        continue  # This key has been exhausted by other threads
                    
                    current_box.append(item)
                    used_items.append((target_key, item))
                    remaining_capacity -= target_key
                    items_count += 1
                    
                    # If perfect packing is achieved, can end early
                    if remaining_capacity == 0 and items_count >= min_items:
                        break
                
                # Check all constraints
                current_capacity = box_capacity - remaining_capacity
                load_ratio = current_capacity / box_capacity
                
                meets_count = items_count >= min_items
                meets_ratio = load_ratio >= min_ratio
                meets_capacity = remaining_capacity >= 0
                
                success = meets_count and meets_ratio and meets_capacity
                
                if success:
                    shared_manager.update_stats('successful_boxes')
                    return True, current_box, thread_id, current_capacity, items_count, f"Success: {items_count} items, load ratio {load_ratio:.1%}"
                else:
                    # Packing failed, rollback
                    shared_manager.rollback_items(used_items)
                    if not meets_count:
                        shared_manager.update_stats('failed_by_count')
                        reason = f"Insufficient items: {items_count}<{min_items}"
                    elif not meets_ratio:
                        shared_manager.update_stats('failed_by_ratio')
                        reason = f"Insufficient load ratio: {load_ratio:.1%}<{min_ratio:.1%}"
                    else:
                        shared_manager.update_stats('failed_by_capacity')
                        reason = "Capacity constraint failed"
                    
                    return False, None, thread_id, current_capacity, items_count, reason
                    
            except Exception as e:
                shared_manager.update_stats('failed_by_capacity')
                return False, None, thread_id, 0, 0, f"Packing exception: {str(e)}"
    
        # 5. Seed preprocessing and screening
        analyzer = SeedPotentialAnalyzer(self.hash_buckets, small_keys, box_capacity, min_items)
        
        # Collect and evaluate all seeds
        seed_candidates = []
        for key in large_keys:
            if key in self.hash_buckets:
                potential = analyzer.calculate_potential(key)
                count = len(self.hash_buckets[key])
                for _ in range(count):
                    seed_candidates.append((key, potential))  # Ensure it's a tuple
        if not seed_candidates:
            self._logger.warning("No available seed candidates")
            return []
        
        # Sort by potential, only process high-potential seeds
        seed_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_threshold = 0.2  # Only process seeds with potential > 0.2
        # high_potential_seeds = [seed for seed, potential in seed_candidates if potential > potential_threshold]
        # Fix: Correctly handle screening logic
        high_potential_candidates = [(seed, potential) for seed, potential in seed_candidates
                                if potential > potential_threshold]

        # Final fallback: keep at least top 50% of seeds
        if len(high_potential_candidates) < len(seed_candidates) * 0.5:
            mid_point = len(seed_candidates) // 2
            high_potential_candidates = seed_candidates[:mid_point]
        
        # Fix: Correctly extract seed list
        selected_seeds = [seed for seed, potential in high_potential_candidates]
        
        self._logger.info(f"Seed screening completed:")
        self._logger.info(f"  Total seeds: {len(seed_candidates)}")
        self._logger.info(f"  After screening: {len(selected_seeds)}")
        self._logger.info(f"  Screening rate: {len(selected_seeds)/len(seed_candidates):.1%}")
        self._logger.info(f"  Thread count: {max_workers}")
        self._logger.info(f"  Constraints: capacity≥{min_ratio:.0%}, items≥{min_items}")
    
        # 6. Initialize shared manager
        shared_manager = EnhancedSharedManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_current_stats()
        
        self._logger.info(f"Initial resource status:")
        self._logger.info(f"  Total small items: {initial_stats['remaining_small_items']}")
        self._logger.info(f"  Available small key types: {initial_stats['available_key_types']}")

        # 7. Multi-threaded packing execution
        output_boxes = []
        detailed_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="ConstraintPack") as executor:
            
            # Submit all tasks
            future_to_seed = {}
            for i, seed_key in enumerate(selected_seeds):
                future = executor.submit(pack_single_seed_with_constraints, seed_key, shared_manager, i)
                future_to_seed[future] = (seed_key, i)
            
            # Process results
            with tqdm(total=len(selected_seeds), unit='seed', 
                     desc='Multi-constraint packing', dynamic_ncols=True) as pbar:
                
                completed_tasks = 0
                for future in as_completed(future_to_seed):
                    seed_key, task_id = future_to_seed[future]
                    
                    try:
                        success, box, thread_id, capacity, item_count, info = future.result(timeout=60)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': success,
                            'capacity': capacity,
                            'item_count': item_count,
                            'info': info,
                            'thread_id': thread_id
                        })
    
                        completed_tasks += 1
                        pbar.update(1)
                        
                        # Update progress description every 100 tasks
                        if completed_tasks % 100 == 0:
                            current_stats = shared_manager.get_current_stats()
                            success_rate = current_stats['successful_boxes'] / max(1, current_stats['total_attempts'])
                            pbar.set_description(
                                f'Multi-constraint packing(Success:{current_stats["successful_boxes"]}, '
                                f'Success rate:{success_rate:.1%}, '
                                f'Remaining:{current_stats["remaining_small_items"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"Task {task_id} (seed={seed_key}) failed: {e}")
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': False,
                            'capacity': 0,
                            'item_count': 0,
                            'info': f"Task exception: {str(e)}",
                            'thread_id': -1
                        })
                        pbar.update(1)
    
        end_time = time.time()

        # 8. Detailed statistical analysis
        final_stats = shared_manager.get_current_stats()
        
        # Classify by failure reason
        failure_analysis = defaultdict(int)
        success_details = []
        
        for result in detailed_results:
            if result['success']:
                success_details.append(result)
            else:
                # Simplify failure reason
                info = result['info']
                if 'Insufficient items' in info:
                    failure_analysis['Insufficient item count'] += 1
                elif 'Insufficient load ratio' in info:
                    failure_analysis['Insufficient load ratio'] += 1
                elif 'No available seed' in info:
                    failure_analysis['Seed exhausted'] += 1
                elif 'Cannot reach' in info:
                    failure_analysis['Feasibility check failed'] += 1
                else:
                    failure_analysis['Other reasons'] += 1

        # 9. Output detailed report
        if output_boxes:
            # Successful packing statistics
            total_items_packed = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items_packed / len(output_boxes)
            capacities = [result['capacity'] for result in success_details]
            avg_capacity = sum(capacities) / len(capacities) if capacities else 0
            avg_load_ratio = avg_capacity / box_capacity
            
            item_counts = [result['item_count'] for result in success_details]
            min_items_in_box = min(item_counts) if item_counts else 0
            max_items_in_box = max(item_counts) if item_counts else 0
            
            self._logger.info(f"Multi-constraint packing completed!")
            self._logger.info(f"Execution statistics:")
            self._logger.info(f"  Total time: {end_time - start_time:.2f}s")
            self._logger.info(f"  Processed seeds: {len(selected_seeds)}")
            self._logger.info(f"  Successful bins: {len(output_boxes)}")
            self._logger.info(f"  Overall success rate: {len(output_boxes)/len(selected_seeds):.2%}")
            
            self._logger.info(f"Packing quality:")
            self._logger.info(f"  Average load ratio: {avg_load_ratio:.1%}")
            self._logger.info(f"  Average item count: {avg_items_per_box:.1f}")
            self._logger.info(f"  Item count range: {min_items_in_box}-{max_items_in_box}")
            self._logger.info(f"  Total packed items: {total_items_packed}")
            
            self._logger.info(f"Remaining resources:")
            self._logger.info(f"  Small items: {final_stats['remaining_small_items']}")
            self._logger.info(f"  Available key types: {final_stats['available_key_types']}")
            
            if failure_analysis:
                self._logger.info(f"Failure analysis:")
                for reason, count in failure_analysis.items():
                    percentage = count / len(selected_seeds) * 100
                    self._logger.info(f"     {reason}: {count} times ({percentage:.1f}%)")
        else:
            self._logger.warning("No items successfully packed!")
            self._logger.info(f"Failure reason distribution: {dict(failure_analysis)}")
            self._logger.info(f"Suggestions:")
            self._logger.info(f"  1. Lower min_items (current: {min_items})")
            self._logger.info(f"  2. Lower min_ratio (current: {min_ratio})")
            self._logger.info(f"  3. Check if data distribution is reasonable")
    
        return output_boxes

    def pack_with_flexible_seeds(self, box_capacity: int = 16384,
                               seed_strategy: str = "auto",
                               seed_params: dict = None,
                               min_items: int = 10, min_ratio: float = 0.95,
                               max_workers: int = None) -> List[np.ndarray]:
        """
        Custom seed selection strategy + item count limit for bins + output bin minimum capacity
        
        Parameters:
            seed_strategy: Seed strategy
                - "auto": Automatically use box_capacity // 2
                - "custom_half": Use a custom half value
                - "specified_keys": Use specified key list
                - "size_range": Use size range filtering
                - "top_n": Use the largest N keys as seeds
                - "capacity_ratio": Specify percentage of capacity to occupy
            seed_params: Strategy parameter dictionary
                - 
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 6)
        
        if seed_params is None:
            seed_params = {}
        
        # Generate seeds based on strategy
        if seed_strategy == "auto":
            half = box_capacity // 2
            large_keys = [k for k in self.hash_buckets.keys() if k >= half]
            
        elif seed_strategy == "custom_half":
            custom_half = seed_params.get("half", box_capacity // 3)
            # max_elems = seed_params.get("n_max", None)         # Maximum number of seeds to extract per key
            large_keys = [k for k in self.hash_buckets.keys() if k >= custom_half]
            
        elif seed_strategy == "specified_keys":
            specified_keys = seed_params.get("keys", [])
            large_keys = [k for k in specified_keys if k in self.hash_buckets]
            
        elif seed_strategy == "size_range":
            min_size = seed_params.get("min_size", box_capacity // 3)
            max_size = seed_params.get("max_size", box_capacity)
            large_keys = [k for k in self.hash_buckets.keys() 
                         if min_size <= k <= max_size]
            
        elif seed_strategy == "top_n":
            n = seed_params.get("n", 5)
            available_keys = sorted(self.hash_buckets.keys(), reverse=True)
            large_keys = available_keys[:n]
            
        elif seed_strategy == "capacity_ratio":
            min_ratio = seed_params.get("min_ratio", 0.3)  # At least 30% capacity
            max_ratio = seed_params.get("max_ratio", 1.0)  # Up to 100% capacity
            min_size = int(box_capacity * min_ratio)
            max_size = int(box_capacity * max_ratio)
            large_keys = [k for k in self.hash_buckets.keys() 
                         if min_size <= k <= max_size]

        
        else:
            raise ValueError(f"Unsupported seed strategy: {seed_strategy}")
        
        # Generate small item list
        small_keys = [k for k in self.hash_buckets.keys() if k not in large_keys]
        
        # Strategy information
        self._logger.info(f"Seed strategy: {seed_strategy}")
        self._logger.info(f"Strategy parameters: {seed_params}")
        self._logger.info(f"  Seed keys(max): {large_keys[-1]}")
        self._logger.info(f"  Filler keys count: {len(small_keys)}")
        
        if not large_keys:
            self._logger.warning(f"Strategy {seed_strategy} did not generate any seeds")
            return []


        # 1. Seed potential analyzer
        class SeedPotentialAnalyzer:
            def __init__(self, hash_buckets, small_keys, box_capacity, min_items):
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.box_capacity = box_capacity
                self.min_items = min_items
                
            def calculate_potential(self, seed_key: int) -> float:
                """Calculate packing success potential for a seed"""
                remaining_capacity = self.box_capacity - seed_key
                
                if remaining_capacity <= 0:
                    return 0.0
                
                # Count distribution of small items
                available_small_items = sum(
                    len(self.hash_buckets[k]) for k in self.small_keys 
                    if k in self.hash_buckets
                )
                
                if available_small_items == 0:
                    return 0.0
                
                # Estimate number of items that can be packed (conservative estimate)
                min_small_key = min(self.small_keys) if self.small_keys else remaining_capacity
                max_possible_items = remaining_capacity // min_small_key
                
                # Consider practical availability (not all small keys have items)
                practical_items = min(max_possible_items, available_small_items // 2)
                total_items = practical_items + 1  # +1 for seed
                
                # Potential score
                count_score = min(total_items / self.min_items, 1.0) if self.min_items > 0 else 1.0
                capacity_score = seed_key / self.box_capacity
                diversity_score = len([k for k in self.small_keys if k <= remaining_capacity]) / len(self.small_keys) if self.small_keys else 0
                
                return count_score * 0.5 + capacity_score * 0.3 + diversity_score * 0.2

        # 2. Enhanced shared resource manager
        class EnhancedSharedManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.large_keys = large_keys
                
                # Maintain statistics of available keys
                self.key_stats = {}
                self._update_key_stats()
                
                # Performance statistics
                self.stats = {
                    'total_attempts': 0,
                    'successful_boxes': 0,
                    'failed_by_count': 0,
                    'failed_by_ratio': 0,
                    'failed_by_capacity': 0
                }
            
            def _update_key_stats(self):
                """Update key statistics"""
                self.key_stats = {}
                for k in self.small_keys:
                    if k in self.hash_buckets and len(self.hash_buckets[k]) > 0:
                        self.key_stats[k] = len(self.hash_buckets[k])
                        
            def get_seed_item(self, seed_key: int) -> Tuple[bool, Optional[np.record]]:
                """Get seed item"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> Tuple[bool, Optional[np.record]]:
                """Get item by specified key"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # Update statistics
                        if target_key in self.key_stats:
                            self.key_stats[target_key] -= 1
                            if self.key_stats[target_key] <= 0:
                                del self.key_stats[target_key]
                        
                        return True, item
                    return False, None

            def get_available_keys_with_counts(self) -> Dict[int, int]:
                """Get available keys and their item counts"""
                with self.lock:
                    return self.key_stats.copy()
            
            def rollback_items(self, items_to_rollback: List[Tuple[int, np.record]]):
                """Rollback items from failed packing attempts"""
                with self.lock:
                    for key, item in reversed(items_to_rollback):  # Rollback in reverse order
                        self.hash_buckets[key] = np.insert(self.hash_buckets[key], 0, item)
                        # Update statistics
                        if key in self.small_keys:
                            self.key_stats[key] = self.key_stats.get(key, 0) + 1
            
            def update_stats(self, result_type: str):
                """Update statistics"""
                with self.lock:
                    self.stats['total_attempts'] += 1
                    if result_type in self.stats:
                        self.stats[result_type] += 1
            
            def get_current_stats(self) -> Dict:
                """Get current statistics"""
                with self.lock:
                    total_small_items = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    return {
                        **self.stats,
                        'remaining_small_items': total_small_items,
                        'available_key_types': len(self.key_stats)
                    }

        # 3. Intelligent packing strategy
        def is_feasible_quick_check(remaining_capacity: int, current_items: int, 
                                   available_keys: Dict[int, int], min_items: int) -> bool:
            """Quick feasibility check"""
            if current_items >= min_items:
                return True
                
            needed_items = min_items - current_items
            if not available_keys:
                return False
            
            # Greedy estimation: prioritize small keys
            sorted_keys = sorted(available_keys.keys())
            possible_items = 0
            remaining_cap = remaining_capacity
            
            for key in sorted_keys:
                if remaining_cap <= 0:
                    break
                max_from_this_key = min(remaining_cap // key, available_keys[key])
                possible_items += max_from_this_key
                remaining_cap -= max_from_this_key * key
                
                if possible_items >= needed_items:
                    return True
                    
            return False
    
        def select_optimal_key(strategy: str, available_keys: Dict[int, int], 
                              remaining_capacity: int, current_items: int, min_items: int) -> Optional[int]:
            """Select optimal key based on strategy"""
            suitable_keys = [k for k in available_keys.keys() 
                            if k <= remaining_capacity and available_keys[k] > 0]
            if not suitable_keys:
                return None
            
            if strategy == "prioritize_count":
                # Prioritize count: select smallest key
                return min(suitable_keys)
            elif strategy == "prioritize_capacity":
                # Prioritize capacity: select largest key
                return max(suitable_keys)
            elif strategy == "balanced":
                # Balanced strategy: select medium size, considering available quantity
                suitable_keys.sort()
                # Prefer keys with more available items
                key_scores = [(k, available_keys[k] * (remaining_capacity / k)) for k in suitable_keys]
                key_scores.sort(key=lambda x: x[1], reverse=True)
                return key_scores[0][0]
            else:
                return suitable_keys[0]

        # 4. Core packing function
        def pack_single_seed_with_constraints(seed_key: int, shared_manager: EnhancedSharedManager, 
                                            thread_id: int) -> Tuple:
            """Single seed packing with constraints"""
            try:
                # Get seed
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    shared_manager.update_stats('failed_by_capacity')
                    return False, None, thread_id, 0, 0, "No available seed"
                
                current_box = [seed_item]
                used_items = [(seed_key, seed_item)]  # For rollback
                remaining_capacity = box_capacity - seed_key
                items_count = 1
                
                max_iterations = min_items * 16  # Prevent infinite loop (changed from 5→15, 12 for 16384)
                iteration = 0
                
                # Main packing loop
                while (remaining_capacity > 0 and 
                       items_count < min_items * 8 and  # Allow exceeding minimum (may have very small values) (5 for 16384)
                       iteration < max_iterations):
                    
                    iteration += 1
                    available_keys = shared_manager.get_available_keys_with_counts()
                    
                    # Quick feasibility check
                    if (items_count < min_items and 
                        not is_feasible_quick_check(remaining_capacity, items_count, 
                                                  available_keys, min_items)):
                        # Cannot reach minimum item count, exit early
                        shared_manager.rollback_items(used_items)
                        shared_manager.update_stats('failed_by_count')
                        return False, None, thread_id, 0, items_count, f"Cannot reach {min_items} items"
                    
                    # Dynamic strategy selection
                    if items_count < min_items * 0.8:
                        strategy = "prioritize_count"
                    elif items_count < min_items:
                        strategy = "balanced"
                    else:
                        strategy = "prioritize_capacity"
                    
                    # Select next key
                    target_key = select_optimal_key(strategy, available_keys, 
                                                  remaining_capacity, items_count, min_items)
                    if target_key is None:
                        break
                    
                    # Get item
                    success, item = shared_manager.get_item_by_key(target_key)
                    if not success:
                        continue  # This key was exhausted by another thread
                    
                    current_box.append(item)
                    used_items.append((target_key, item))
                    remaining_capacity -= target_key
                    items_count += 1
                    
                    # If perfect packing achieved, end early
                    if remaining_capacity == 0 and items_count >= min_items:
                        break
                
                # Check all constraints
                current_capacity = box_capacity - remaining_capacity
                load_ratio = current_capacity / box_capacity
                
                meets_count = items_count >= min_items
                meets_ratio = load_ratio >= min_ratio
                meets_capacity = remaining_capacity >= 0
                
                success = meets_count and meets_ratio and meets_capacity
                
                if success:
                    shared_manager.update_stats('successful_boxes')
                    return True, current_box, thread_id, current_capacity, items_count, f"Success: {items_count} items, load rate {load_ratio:.1%}"
                else:
                    # Packing failed, rollback
                    shared_manager.rollback_items(used_items)
                    if not meets_count:
                        shared_manager.update_stats('failed_by_count')
                        reason = f"Insufficient items: {items_count}<{min_items}"
                    elif not meets_ratio:
                        shared_manager.update_stats('failed_by_ratio')
                        reason = f"Insufficient load rate: {load_ratio:.1%}<{min_ratio:.1%}"
                    else:
                        shared_manager.update_stats('failed_by_capacity')
                        reason = "Capacity constraint failed"
                    
                    return False, None, thread_id, current_capacity, items_count, reason
                    
            except Exception as e:
                shared_manager.update_stats('failed_by_capacity')
                return False, None, thread_id, 0, 0, f"Packing exception: {str(e)}"

        # 5. Seed preprocessing and filtering
        analyzer = SeedPotentialAnalyzer(self.hash_buckets, small_keys, box_capacity, min_items)
        
        # Collect and evaluate all seeds
        seed_candidates = []
        for key in large_keys:
            if key in self.hash_buckets:
                potential = analyzer.calculate_potential(key)
                count = len(self.hash_buckets[key])
                for _ in range(count):
                    seed_candidates.append((key, potential))  # Ensure it's a tuple
        if not seed_candidates:
            self._logger.warning("No available seed candidates")
            return []
        
        # Sort by potential, only process high-potential seeds
        seed_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_threshold = 0.2  # Only process seeds with potential > 0.2
        # high_potential_seeds = [seed for seed, potential in seed_candidates if potential > potential_threshold]
        # Fix: Properly handle filtering logic
        high_potential_candidates = [(seed, potential) for seed, potential in seed_candidates 
                                if potential > potential_threshold]

        # Final fallback: keep at least top 50% of seeds
        if len(high_potential_candidates) < len(seed_candidates) * 0.5:
            mid_point = len(seed_candidates) // 2
            high_potential_candidates = seed_candidates[:mid_point]
        
        # Fix: Correctly extract seed list
        selected_seeds = [seed for seed, potential in high_potential_candidates]
        
        self._logger.info(f"Seed filtering completed:")
        self._logger.info(f"  Total seeds: {len(seed_candidates)}")
        self._logger.info(f"  After filtering: {len(selected_seeds)}")
        self._logger.info(f"  Filtering rate: {len(selected_seeds)/len(seed_candidates):.1%}")
        self._logger.info(f"  Thread count: {max_workers}")
        self._logger.info(f"  Constraints: Capacity≥{min_ratio:.0%}, Items≥{min_items}")
    
        # 6. Initialize shared manager
        shared_manager = EnhancedSharedManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_current_stats()
        
        self._logger.info(f"Initial resource status:")
        self._logger.info(f"  Total small items: {initial_stats['remaining_small_items']}")
        self._logger.info(f"  Available small key types: {initial_stats['available_key_types']}")

        # 7. Multi-threaded packing execution
        output_boxes = []
        detailed_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="ConstraintPack") as executor:
            
            # Submit all tasks
            future_to_seed = {}
            for i, seed_key in enumerate(selected_seeds):
                future = executor.submit(pack_single_seed_with_constraints, seed_key, shared_manager, i)
                future_to_seed[future] = (seed_key, i)
            
            # Process results
            with tqdm(total=len(selected_seeds), unit='seed', 
                     desc='Multi-constraint packing', dynamic_ncols=True) as pbar:
                
                completed_tasks = 0
                for future in as_completed(future_to_seed):
                    seed_key, task_id = future_to_seed[future]
                    
                    try:
                        success, box, thread_id, capacity, item_count, info = future.result(timeout=60)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': success,
                            'capacity': capacity,
                            'item_count': item_count,
                            'info': info,
                            'thread_id': thread_id
                        })
    
                        completed_tasks += 1
                        pbar.update(1)
                        
                        # Update progress description every 100 tasks
                        if completed_tasks % 100 == 0:
                            current_stats = shared_manager.get_current_stats()
                            success_rate = current_stats['successful_boxes'] / max(1, current_stats['total_attempts'])
                            pbar.set_description(
                                f'Multi-constraint packing(Success:{current_stats["successful_boxes"]}, '
                                f'Success rate:{success_rate:.1%}, '
                                f'Remaining:{current_stats["remaining_small_items"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"Task {task_id} (seed={seed_key}) failed: {e}")
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': False,
                            'capacity': 0,
                            'item_count': 0,
                            'info': f"Task exception: {str(e)}",
                            'thread_id': -1
                        })
                        pbar.update(1)
    
        end_time = time.time()

        # 8. Detailed statistical analysis
        final_stats = shared_manager.get_current_stats()
        
        # Classify by failure reason
        failure_analysis = defaultdict(int)
        success_details = []
        
        for result in detailed_results:
            if result['success']:
                success_details.append(result)
            else:
                # Simplify failure reason
                info = result['info']
                if 'Insufficient items' in info:
                    failure_analysis['Insufficient item count'] += 1
                elif 'Insufficient load rate' in info:
                    failure_analysis['Insufficient load rate'] += 1
                elif 'No available seed' in info:
                    failure_analysis['Seed exhausted'] += 1
                elif 'Cannot reach' in info:
                    failure_analysis['Feasibility check failed'] += 1
                else:
                    failure_analysis['Other reasons'] += 1

        # 9. Output detailed report
        if output_boxes:
            # Successful packing statistics
            total_items_packed = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items_packed / len(output_boxes)
            capacities = [result['capacity'] for result in success_details]
            avg_capacity = sum(capacities) / len(capacities) if capacities else 0
            avg_load_ratio = avg_capacity / box_capacity
            
            item_counts = [result['item_count'] for result in success_details]
            min_items_in_box = min(item_counts) if item_counts else 0
            max_items_in_box = max(item_counts) if item_counts else 0
            
            self._logger.info(f"Multi-constraint packing completed!")
            self._logger.info(f"Execution statistics:")
            self._logger.info(f"  Total time: {end_time - start_time:.2f}s")
            self._logger.info(f"  Seeds processed: {len(selected_seeds)}")
            self._logger.info(f"  Successful bins: {len(output_boxes)}")
            self._logger.info(f"  Overall success rate: {len(output_boxes)/len(selected_seeds):.2%}")
            
            self._logger.info(f"Packing quality:")
            self._logger.info(f"  Average load rate: {avg_load_ratio:.1%}")
            self._logger.info(f"  Average items per bin: {avg_items_per_box:.1f}")
            self._logger.info(f"  Item count range: {min_items_in_box}-{max_items_in_box}")
            self._logger.info(f"  Total packed items: {total_items_packed}")
            
            self._logger.info(f"Remaining resources:")
            self._logger.info(f"  Small items: {final_stats['remaining_small_items']}")
            self._logger.info(f"  Available key types: {final_stats['available_key_types']}")
            
            if failure_analysis:
                self._logger.info(f"Failure analysis:")
                for reason, count in failure_analysis.items():
                    percentage = count / len(selected_seeds) * 100
                    self._logger.info(f"     {reason}: {count} times ({percentage:.1f}%)")
        else:
            self._logger.warning("No items successfully packed!")
            self._logger.info(f"Failure reason distribution: {dict(failure_analysis)}")
            self._logger.info(f"Suggestions:")
            self._logger.info(f"  1. Lower min_items (current: {min_items})")
            self._logger.info(f"  2. Lower min_ratio (current: {min_ratio})")
            self._logger.info(f"  3. Check if data distribution is reasonable")


        
        # Return only 1 for status tracking, return 3 for actual application
        return output_boxes#, failure_analysis, final_stats

    def pack_simplest_strategy(
        self,
        keys: List[int],
        m: int,
        box_capacity: int = 16384,
        min_ratio: float = 0.95,
        max_workers: int = None,
    ) -> List[np.ndarray]:
        """
        Simplest packing strategy:
        1. Randomly select m seeds from the specified keys;
        2. All remaining elements form the filling pool;
        3. Multi-threaded packing (delete on success, rollback on failure);
        4. Single-threaded bottom line for remaining elements, the last batch is forced to output and cleared.
        """
        import random
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if max_workers is None:
            max_workers = min(os.cpu_count(), 8)

        # ---------- 1. Construct seed pool & filling pool ----------
        seed_pool = []           # [(key, item), ...]
        fill_buckets = defaultdict(list)   # key -> [item, ...]

        # 1.1 Collect seed pool & unselected elements of the same key
        for k in keys:
            if k not in self.hash_buckets or len(self.hash_buckets[k]) == 0:
                continue
            arr = self.hash_buckets[k]
            # Randomly select m, select all if insufficient
            chosen = random.sample(list(arr), min(m, len(arr)))
            seed_pool.extend([(k, item) for item in chosen])
            # Unselected go into the filling pool
            mask = np.ones(len(arr), dtype=bool)
            idxs = [i for i, it in enumerate(arr) if it in chosen]
            mask[idxs] = False
            fill_buckets[k].extend(arr[mask])

        # 1.2 All elements not in keys are put into the filling pool
        for k in self.hash_buckets:
            if k not in keys:
                fill_buckets[k].extend(self.hash_buckets[k])

        if not seed_pool:
            self._logger.warning("Seed pool is empty, directly output the remaining elements as one box")
            # Force output one box
            leftover = []
            for k, items in fill_buckets.items():
                leftover.extend(items)
            if leftover:
                box = np.array(leftover, dtype=self.DTYPE_SAMPLE_INFO)
                # Clear hash_buckets
                for k in list(self.hash_buckets.keys()):
                    del self.hash_buckets[k]
                return [box]
            return []

        # ---------- 2. Thread-safe resource manager ----------
        class SimpleManager:
            def __init__(self, seed_items, fill_dict, dtype):
                self.lock = threading.RLock()
                # Seed queue
                self.seed_q = seed_items[:]          # Copy
                # Filling pool
                self.fill = defaultdict(deque)
                for k, lst in fill_dict.items():
                    self.fill[k] = deque(lst)
                # Statistics
                self.boxes = []
                self.attempts = 0
                self.success = 0
                self.dtype = dtype

            def pop_seed(self):
                with self.lock:
                    if not self.seed_q:
                        return None
                    return self.seed_q.pop()

            def pop_fill(self, key):
                with self.lock:
                    if not self.fill[key]:
                        return None
                    return self.fill[key].popleft()

            def add_box(self, box):
                with self.lock:
                    self.boxes.append(np.array(box, dtype=self.dtype))
                    self.success += 1

            def rollback(self, rollback_items):
                with self.lock:
                    for key, item in reversed(rollback_items):
                        self.fill[key].appendleft(item)

            def remaining_elements(self):
                with self.lock:
                    return sum(len(q) for q in self.fill.values())

            def all_items(self):
                with self.lock:
                    items = []
                    for k, q in self.fill.items():
                        items.extend(q)
                    return items

        from collections import deque
        # mgr = SimpleManager(seed_pool, fill_buckets)
        mgr = SimpleManager(seed_pool, fill_buckets, self.DTYPE_SAMPLE_INFO)

        # ---------- 3. Multi-threaded packing ----------
        def pack_once(args):
            seed_key, seed_item, tid = args
            box = [seed_item]
            used = [(seed_key, seed_item)]
            rem = box_capacity - seed_key

            # Greedy filling
            for k in sorted(mgr.fill.keys(), reverse=True):
                while rem >= k and mgr.fill[k]:
                    it = mgr.pop_fill(k)
                    if it is None:
                        break
                    box.append(it)
                    used.append((k, it))
                    rem -= k
                    if rem == 0:
                        break

            load = box_capacity - rem
            if load >= min_ratio * box_capacity:
                mgr.add_box(box)
                return True, tid, load
            else:
                mgr.rollback(used)
                return False, tid, load

        # Construct task list
        tasks = [(k, it, i) for i, (k, it) in enumerate(mgr.seed_q)]
        mgr.seed_q.clear()   # Clear, replaced by task list

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futs = [exe.submit(pack_once, t) for t in tasks]
            for f in as_completed(futs):
                ok, tid, load = f.result()
                mgr.attempts += 1

        # ---------- 4. Single-threaded bottom line ----------
        leftover_keys = list(mgr.fill.keys())
        random.shuffle(leftover_keys)

        while mgr.remaining_elements() > 0:
            # Randomly find a seed: pick an element from the remaining keys
            candidates = [(k, mgr.fill[k][0]) for k in leftover_keys if mgr.fill[k]]
            if not candidates:
                break
            seed_key, seed_item = random.choice(candidates)
            mgr.pop_fill(seed_key)  # Take out as seed

            box = [seed_item]
            used = [(seed_key, seed_item)]
            rem = box_capacity - seed_key

            # Continue filling
            for k in sorted(mgr.fill.keys(), reverse=True):
                while rem >= k and mgr.fill[k]:
                    it = mgr.pop_fill(k)
                    if it is None:
                        break
                    box.append(it)
                    used.append((k, it))
                    rem -= k
                    if rem == 0:
                        break

            # Force output
            mgr.add_box(box)

        # ---------- 5. Sync back to self.hash_buckets ----------
        # At this point, mgr.fill has been completely cleared, so clear hash_buckets directly
        for k in list(self.hash_buckets.keys()):
            del self.hash_buckets[k]

        self._logger.info(
            f"pack_simplest_strategy completed: multi-threaded tasks {mgr.attempts}, "
            f"successful {mgr.success}, bottom line output {len(mgr.boxes) - mgr.success} boxes"
        )
        return mgr.boxes



    
    def check_hash_buckets_state(self):
        """Check the current state of the hash buckets"""
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        # total_keys = len(self.hash_buckets)
        total_keys = len([key for key in self.hash_buckets if len(self.hash_buckets[key])>0])  # Do not delete keys with zero elements
        
        # Classify and count by key size
        key_distribution = defaultdict(int)
        for key in self.hash_buckets.keys():
            if key >= 8192:
                key_distribution['large'] += len(self.hash_buckets[key])
            elif key >= 2048:
                key_distribution['medium'] += len(self.hash_buckets[key])
            else:
                key_distribution['small'] += len(self.hash_buckets[key])
        
        print(f"Current hash buckets state:")
        print(f"  Total items: {total_items}")
        print(f"  Total keys: {total_keys}")
        print(f"  Distribution:")
        for size, count in key_distribution.items():
            print(f"    {size}: {count} items")
        
        return {
            'total_items': total_items,
            'total_keys': total_keys,
            'key_distribution': dict(key_distribution)
        }


class PackingTracker:
    """Packing operation tracker"""
    def __init__(self, processor):
        self.processor = processor
        self.history = []
        
    def track_packing(self, strategy_name: str, **kwargs):
        """Record a packing operation"""
        before_state = self.processor.check_hash_buckets_state()
        # Supports returning detailed statistics (e.g., total_attempts), otherwise returns only the box list
        result = getattr(self.processor, strategy_name)(**kwargs)
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
            boxes = result[0]
            stats = result[1]
            total_attempts = stats.get('total_attempts', len(boxes))
        else:
            boxes = result
            total_attempts = len(boxes)
        after_state = self.processor.check_hash_buckets_state()
        change = {
            'strategy': strategy_name,
            'kwargs': kwargs,
            'before': before_state,
            'after': after_state,
            'boxes_count': len(boxes),
            'items_used': before_state['total_items'] - after_state['total_items'],
            'total_attempts': total_attempts
        }
        self.history.append(change)
        return boxes
    
    def print_summary(self):
        """Print packing history summary"""
        print("\n=== Packing Operation History ===")
        for i, op in enumerate(self.history, 1):
            print(f"\nOperation {i}: {op['strategy']}")
            print(f"Parameters: {op['kwargs']}")
            print(f"Boxes count: {op['boxes_count']}")
            print(f"Items used: {op['items_used']}")
            if op.get('total_attempts', 0):
                rate = op['boxes_count'] / op['total_attempts']
                print(f"Success rate: {rate:.1%} ({op['boxes_count']}/{op['total_attempts']})")
            else:
                print(f"Success rate: N/A")


def analyze_packing_history(tracker):
    """Analyze packing history"""
    print("\n=== Detailed Analysis ===")
    
    total_boxes = sum(op['boxes_count'] for op in tracker.history)
    total_items = sum(op['items_used'] for op in tracker.history)
    
    print(f"Total boxes: {total_boxes}")
    print(f"Total items used: {total_items}")
    
    # Analyze the effectiveness of each strategy
    strategy_stats = defaultdict(lambda: {'count': 0, 'items': 0, 'boxes': 0})
    for op in tracker.history:
        strategy = op['strategy']
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['items'] += op['items_used']
        strategy_stats[strategy]['boxes'] += op['boxes_count']
    
    print("\nStrategy performance comparison:")
    for strategy, stats in strategy_stats.items():
        avg_boxes = stats['boxes'] / stats['count']
        avg_items = stats['items'] / stats['count']
        print(f"{strategy}:")
        print(f"  Average boxes: {avg_boxes:.1f}")
        print(f"  Average items used: {avg_items:.0f}")
        print(f"  Average success rate: {avg_boxes / 4:.1%}")  # Assuming 4 threads are used

import pickle

def save_ckpt(tracker, file_path: str):
    """
    Save tracker (including processor) to file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(tracker, f)
    print(f"Checkpoint saved to {file_path}")

def load_ckpt(file_path: str):
    """
    Load tracker (including processor) state
    """
    with open(file_path, 'rb') as f:
        tracker = pickle.load(f)
    print(f"Checkpoint loaded: {file_path}")
    return tracker

def save_bin_boxes(bin_boxes, file_path: str):
    """
    Save single-step packing result
    """
    with open(file_path, 'wb') as f:
        pickle.dump(bin_boxes, f)
    print(f"Packing result saved to {file_path}")

def load_bin_boxes(file_path: str):
    """
    Load single-step packing result
    """
    with open(file_path, 'rb') as f:
        bin_boxes = pickle.load(f)
    print(f"Packing result loaded: {file_path}")
    return bin_boxes