from typing import Iterator, Dict, Any, List
import numpy as np

from .parser import parse_sql
from .logical_plan import (
    build_logical_plan, LogicalPlanNode, LogicalScan, LogicalFilter,
    LogicalProject, LogicalJoin, LogicalAggregate, LogicalSort
)
from .optimizer import Optimizer
from .physical_plan import (
    Chunk, PhysicalOperator, PhysicalScan, PhysicalFilter, PhysicalProject,
    PhysicalHashJoin, PhysicalSortMergeJoin, PhysicalGroupByAggregate, PhysicalOrderBy
)
from .ast_nodes import Expr, Column, BinaryOp, Literal, SortItem

class QueryContext:
    def __init__(self, data_sources: Dict[str, str]):
        # maps table_name -> file_path
        self.data_sources = data_sources

def compile_physical_plan(node: LogicalPlanNode, ctx: QueryContext) -> PhysicalOperator:
    if isinstance(node, LogicalScan):
        if node.table_name not in ctx.data_sources:
            raise ValueError(f"Table {node.table_name} not found in catalog")
        file_path = ctx.data_sources[node.table_name]
        return PhysicalScan(file_path, node.columns)
        
    elif isinstance(node, LogicalFilter):
        child_op = compile_physical_plan(node.child, ctx)
        return PhysicalFilter(child_op, node.predicate)
        
    elif isinstance(node, LogicalProject):
        child_op = compile_physical_plan(node.child, ctx)
        aliases = []
        new_exprs = []
        from .ast_nodes import AggFunc
        for expr in node.exprs:
            if isinstance(expr, AggFunc):
                name = expr.alias or f"agg_{id(expr)}"
                expr.alias = name
                aliases.append(name)
                new_exprs.append(Column(name))
            elif hasattr(expr, 'name'):
                aliases.append(expr.name)
                new_exprs.append(expr)
            else:
                aliases.append(f"expr_{id(expr)}")
                new_exprs.append(expr)
        return PhysicalProject(child_op, new_exprs, aliases)
        
    elif isinstance(node, LogicalJoin):
        left_op = compile_physical_plan(node.left, ctx)
        right_op = compile_physical_plan(node.right, ctx)
        
        # We assume an equijoin and extract keys
        if not (isinstance(node.on_expr, BinaryOp) and node.on_expr.op == '='):
            raise NotImplementedError("Only equijoins supported")
            
        # Left and right might be swapped based on syntax, but physical operators 
        # need left_key corresponding to left side, right_key to right side. 
        # In this simple implementation we just pass them directly, ASSUMING 
        # the user query orders LHS = RHS properly matching table sides.
        left_key = node.on_expr.left
        right_key = node.on_expr.right
        
        # Use Hash Join by default, could use Sort Merge Join based on metadata
        # Given no stats, let's use Hash Join
        return PhysicalHashJoin(left_op, right_op, left_key, right_key, node.join_type)
        
    elif isinstance(node, LogicalAggregate):
        child_op = compile_physical_plan(node.child, ctx)
        aggs_tuples = [(agg.expr, agg.func, agg.alias or f"agg_{id(agg)}") for agg in node.aggs]
        return PhysicalGroupByAggregate(child_op, node.group_by, aggs_tuples)
        
    elif isinstance(node, LogicalSort):
        child_op = compile_physical_plan(node.child, ctx)
        order_exprs = [(item.expr, item.desc) for item in node.order_by]
        return PhysicalOrderBy(child_op, order_exprs)
        
    raise NotImplementedError(f"Cannot compile {type(node)}")

import hashlib
import os

_QUERY_CACHE = {}

def execute_query(sql: str, ctx: QueryContext) -> List[Chunk]:
    mtimes = []
    for path in sorted(ctx.data_sources.values()):
        if os.path.exists(path):
            mtimes.append(str(os.path.getmtime(path)))
            
    hash_payload = sql + "|" + "|".join(mtimes)
    cache_key = hashlib.md5(hash_payload.encode('utf-8')).hexdigest()
    
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    ast = parse_sql(sql)
    logical_plan = build_logical_plan(ast)
    optimized_plan = Optimizer.optimize(logical_plan)
    physical_plan = compile_physical_plan(optimized_plan, ctx)
    
    chunks = []
    for chunk in physical_plan.execute():
        chunks.append(chunk)
        
    _QUERY_CACHE[cache_key] = chunks
    return chunks
