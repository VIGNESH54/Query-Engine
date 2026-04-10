import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
from typing import Iterator, Dict, List, Any, Optional, Tuple
from .ast_nodes import Expr, Column, BinaryOp, Literal

Chunk = Dict[str, np.ndarray]

def evaluate_expr(expr: Expr, chunk: Chunk) -> np.ndarray:
    if isinstance(expr, Column):
        return chunk[expr.name]
    elif isinstance(expr, Literal):
        if not chunk:
            return np.array([expr.value])
        length = len(next(iter(chunk.values())))
        if isinstance(expr.value, str):
            return np.full(length, expr.value, dtype=object)
        return np.full(length, expr.value)
    elif isinstance(expr, BinaryOp):
        left_vals = evaluate_expr(expr.left, chunk)
        right_vals = evaluate_expr(expr.right, chunk)
        
        if expr.op == '=': return left_vals == right_vals
        elif expr.op == '>': return left_vals > right_vals
        elif expr.op == '<': return left_vals < right_vals
        elif expr.op == '>=': return left_vals >= right_vals
        elif expr.op == '<=': return left_vals <= right_vals
        elif expr.op == '!=': return left_vals != right_vals
        elif expr.op == 'AND': return left_vals & right_vals
        elif expr.op == 'OR': return left_vals | right_vals
        elif expr.op == '+': return left_vals + right_vals
        elif expr.op == '-': return left_vals - right_vals
        elif expr.op == '*': return left_vals * right_vals
        elif expr.op == '/': return left_vals / right_vals
            
    raise NotImplementedError(f"Cannot evaluate {expr}")

class PhysicalOperator:
    def execute(self) -> Iterator[Chunk]:
        raise NotImplementedError

class PhysicalScan(PhysicalOperator):
    def __init__(self, file_path: str, columns: Optional[List[str]]):
        self.file_path = file_path
        self.columns = columns
        
    def execute(self) -> Iterator[Chunk]:
        BATCH_SIZE = 65536
        if self.file_path.endswith('.parquet'):
            parquet_file = pq.ParquetFile(self.file_path)
            # If columns is empty list, implies no projection pushdown is strict, just read all
            cols_to_read = self.columns if self.columns else None
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=cols_to_read):
                chunk = {}
                for idx, col_name in enumerate(batch.schema.names):
                    chunk[col_name] = batch.column(idx).to_numpy(zero_copy_only=False)
                yield chunk
        elif self.file_path.endswith('.csv'):
            table = pcsv.read_csv(self.file_path)
            if self.columns:
                table = table.select(self.columns)
            num_rows = table.num_rows
            for start_idx in range(0, num_rows, BATCH_SIZE):
                batch = table.slice(start_idx, BATCH_SIZE)
                chunk = {}
                for idx, col_name in enumerate(batch.schema.names):
                    chunk[col_name] = batch.column(idx).to_numpy(zero_copy_only=False)
                yield chunk

class PhysicalProject(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, exprs: List[Expr], aliases: List[str]):
        self.child = child
        self.exprs = exprs
        self.aliases = aliases
        
    def execute(self) -> Iterator[Chunk]:
        for chunk in self.child.execute():
            out_chunk = {}
            for expr, alias in zip(self.exprs, self.aliases):
                out_chunk[alias] = evaluate_expr(expr, chunk)
            yield out_chunk

class PhysicalFilter(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, predicate: Expr):
        self.child = child
        self.predicate = predicate
        
    def execute(self) -> Iterator[Chunk]:
        for chunk in self.child.execute():
            mask = evaluate_expr(self.predicate, chunk)
            if mask.any():
                out_chunk = {}
                for k, v in chunk.items():
                    out_chunk[k] = v[mask]
                yield out_chunk

def _concat_chunks(chunks: List[Chunk]) -> Chunk:
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {k: np.concatenate([c[k] for c in chunks]) for k in keys}

class PhysicalHashJoin(PhysicalOperator):
    def __init__(self, build_side: PhysicalOperator, probe_side: PhysicalOperator, 
                 build_key: Expr, probe_key: Expr, join_type: str = 'INNER'):
        self.build_side = build_side
        self.probe_side = probe_side
        self.build_key = build_key
        self.probe_key = probe_key
        self.join_type = join_type

    def execute(self) -> Iterator[Chunk]:
        build_chunks = list(self.build_side.execute())
        if not build_chunks:
            return  # Empty build side
            
        build_full = _concat_chunks(build_chunks)
        build_keys_arr = evaluate_expr(self.build_key, build_full)
        
        # Build hash table tracking indices for each key
        # In NumPy, a good way without pure Python dicts is pandas/factorize, but strictly numpy:
        # We can sort to group, but then it's a sort merge join!
        # A purely numpy hash map is hard, so we'll use a python dict mapping key -> array of indices
        # or use np.unique
        unique_keys, inverse_idx, counts = np.unique(build_keys_arr, return_inverse=True, return_counts=True)
        # Array of indices sorting the keys
        sort_idx = np.argsort(inverse_idx)
        
        # Probe side
        for chunk in self.probe_side.execute():
            probe_keys_arr = evaluate_expr(self.probe_key, chunk)
            
            # Find indices of probe keys in unique_keys
            # np.searchsorted requires sorted unique keys, which np.unique provides
            idx_in_unique = np.searchsorted(unique_keys, probe_keys_arr)
            # Mask valid matches
            valid_mask = (idx_in_unique < len(unique_keys)) & (unique_keys[np.minimum(idx_in_unique, len(unique_keys)-1)] == probe_keys_arr)
            
            match_unique_idx = idx_in_unique[valid_mask]
            probe_row_idx = np.arange(len(probe_keys_arr))[valid_mask]
            
            # Now we must map from match_unique_idx to all matching build_row_idx
            # We can use the counts to find offsets into sort_idx
            offsets = np.zeros(len(counts) + 1, dtype=int)
            np.cumsum(counts, out=offsets[1:])
            
            starts = offsets[match_unique_idx]
            ends = offsets[match_unique_idx + 1]
            lengths = ends - starts
            
            if np.all(lengths == 1):
                out_probe_idx = probe_row_idx
                out_build_idx = sort_idx[starts]
            else:
                out_probe_idx = np.repeat(probe_row_idx, lengths)
                if len(starts) > 0:
                    out_build_idx = np.concatenate([sort_idx[s:e] for s, e in zip(starts, ends)])
                else:
                    out_build_idx = np.array([], dtype=int)
                
            if len(out_probe_idx) > 0:
                out_chunk = {}
                for k, v in chunk.items():
                    out_chunk[k] = v[out_probe_idx]
                for k, v in build_full.items():
                    out_chunk[k] = v[out_build_idx]
                yield out_chunk

class PhysicalGroupByAggregate(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, group_by_exprs: List[Expr], aggs: List[Tuple[Expr, str, str]]):
        # aggs: list of (col_expr, agg_func, alias)
        self.child = child
        self.group_by_exprs = group_by_exprs
        self.aggs = aggs

    def execute(self) -> Iterator[Chunk]:
        child_chunks = list(self.child.execute())
        if not child_chunks:
            return
            
        full_chunk = _concat_chunks(child_chunks)
        
        # Grouping
        if self.group_by_exprs:
            group_arrays = [evaluate_expr(e, full_chunk) for e in self.group_by_exprs]
            # Use lexsort to sort by groups, or np.unique(return_inverse=True) on multiple columns
            # For multiple columns in np.unique, we can stack them as a structured array or use pandas
            # Since numpy unique doesn't support multiple arrays natively without acrobatics:
            # We convert to a single string array or structured array.
            if len(group_arrays) == 1:
                unique_vals, inverse_idx = np.unique(group_arrays[0], return_inverse=True)
                group_vals = [unique_vals]
            else:
                # Fallback to string concatenation for grouping key serialization
                str_arrs = [arr.astype(str) for arr in group_arrays]
                joined_keys = np.char.add(str_arrs[0], "|")
                for arr in str_arrs[1:]:
                    joined_keys = np.char.add(joined_keys, arr)
                    joined_keys = np.char.add(joined_keys, "|")
                unique_keys, inverse_idx = np.unique(joined_keys, return_index=True, return_inverse=True)
                group_vals = [arr[unique_keys_idx] for arr, unique_keys_idx in zip(group_arrays, unique_keys)]
        else:
            inverse_idx = np.zeros(len(next(iter(full_chunk.values()))), dtype=int)
            group_vals = []
            
        num_groups = inverse_idx.max() + 1
        out_chunk = {}
        
        for e, arrs in zip(self.group_by_exprs, group_vals):
            alias = e.name if hasattr(e, 'name') else 'group_key'
            out_chunk[alias] = arrs
            
        for agg_expr, func, alias in self.aggs:
            # Func could be sum, count, min, max
            data = evaluate_expr(agg_expr, full_chunk)
            
            if func == 'SUM':
                res = np.bincount(inverse_idx, weights=data)
            elif func == 'COUNT':
                res = np.bincount(inverse_idx)
            elif func == 'MIN' or func == 'MAX':
                # No native bincount for min/max, implement using sort
                # Sort the data by groups
                sort_idx = np.argsort(inverse_idx)
                sorted_groups = inverse_idx[sort_idx]
                sorted_data = data[sort_idx]
                
                # Find group boundaries
                group_changes = np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1]))
                starts = np.where(group_changes)[0]
                ends = np.concatenate((starts[1:], [len(sorted_groups)]))
                
                res = np.zeros(num_groups)
                if func == 'MAX':
                    np.maximum.reduceat(sorted_data, starts, out=res)
                else:
                    np.minimum.reduceat(sorted_data, starts, out=res)
            else:
                raise NotImplementedError(func)
                
            out_chunk[alias] = res
            
        yield out_chunk

class PhysicalOrderBy(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, order_by_exprs: List[Tuple[Expr, bool]]):
        self.child = child
        self.order_by_exprs = order_by_exprs # list of (expr, descending)

    def execute(self) -> Iterator[Chunk]:
        child_chunks = list(self.child.execute())
        if not child_chunks:
            return
            
        full_chunk = _concat_chunks(child_chunks)
        
        # We need to sort by multiple columns
        # np.lexsort sorts by keys where the LAST key is the primary sort key.
        # So we reverse the order_by_exprs loop
        keys = []
        for expr, desc in reversed(self.order_by_exprs):
            val = evaluate_expr(expr, full_chunk)
            if desc:
                # numeric flip for descending
                if np.issubdtype(val.dtype, np.number):
                    val = -val
                else:
                    raise NotImplementedError("Descending sort on strings not fully supported by simple negation")
            keys.append(val)
            
        sort_idx = np.lexsort(keys)
        out_chunk = {k: v[sort_idx] for k, v in full_chunk.items()}
        yield out_chunk

class PhysicalSortMergeJoin(PhysicalOperator):
    def __init__(self, left_side: PhysicalOperator, right_side: PhysicalOperator, 
                 left_key: Expr, right_key: Expr):
        self.left_side = left_side
        self.right_side = right_side
        self.left_key = left_key
        self.right_key = right_key

    def execute(self) -> Iterator[Chunk]:
        left_chunks = list(self.left_side.execute())
        right_chunks = list(self.right_side.execute())
        if not left_chunks or not right_chunks:
            return
            
        left_full = _concat_chunks(left_chunks)
        right_full = _concat_chunks(right_chunks)
        
        l_keys = evaluate_expr(self.left_key, left_full)
        r_keys = evaluate_expr(self.right_key, right_full)
        
        # Sort both sides
        l_sort_idx = np.argsort(l_keys)
        r_sort_idx = np.argsort(r_keys)
        
        l_keys_sorted = l_keys[l_sort_idx]
        r_keys_sorted = r_keys[r_sort_idx]
        
        # Merge cursors
        out_l_idx = []
        out_r_idx = []
        
        l_pos = 0
        r_pos = 0
        l_len = len(l_keys_sorted)
        r_len = len(r_keys_sorted)
        
        while l_pos < l_len and r_pos < r_len:
            lk = l_keys_sorted[l_pos]
            rk = r_keys_sorted[r_pos]
            
            if lk < rk:
                l_pos += 1
            elif lk > rk:
                r_pos += 1
            else:
                l_end = l_pos
                while l_end < l_len and l_keys_sorted[l_end] == lk:
                    l_end += 1
                    
                r_end = r_pos
                while r_end < r_len and r_keys_sorted[r_end] == rk:
                    r_end += 1
                    
                for i in range(l_pos, l_end):
                    for j in range(r_pos, r_end):
                        out_l_idx.append(l_sort_idx[i])
                        out_r_idx.append(r_sort_idx[j])
                        
                l_pos = l_end
                r_pos = r_end
                
        if out_l_idx:
            out_chunk = {}
            for k, v in left_full.items():
                out_chunk[k] = v[np.array(out_l_idx, dtype=np.int64)]
            for k, v in right_full.items():
                col_name = k if k not in out_chunk else k + '_right'
                out_chunk[col_name] = v[np.array(out_r_idx, dtype=np.int64)]
            yield out_chunk
