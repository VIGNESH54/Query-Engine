from typing import List, Set
from .ast_nodes import Expr, Column, BinaryOp, Literal
from .logical_plan import (
    LogicalPlanNode, LogicalScan, LogicalFilter, 
    LogicalProject, LogicalJoin, LogicalAggregate, LogicalSort
)

def extract_columns(expr: Expr) -> Set[str]:
    cols = set()
    if isinstance(expr, Column):
        cols.add(expr.name)
    elif isinstance(expr, BinaryOp):
        cols.update(extract_columns(expr.left))
        cols.update(extract_columns(expr.right))
    return cols

class Optimizer:
    @staticmethod
    def optimize(plan: LogicalPlanNode) -> LogicalPlanNode:
        plan = Optimizer.pushdown_predicates(plan)
        plan = Optimizer.prune_columns(plan)
        return plan

    @staticmethod
    def pushdown_predicates(node: LogicalPlanNode) -> LogicalPlanNode:
        if isinstance(node, LogicalFilter):
            child = Optimizer.pushdown_predicates(node.child)
            if isinstance(child, LogicalJoin):
                cols = extract_columns(node.predicate)
                return LogicalFilter(child=child, predicate=node.predicate)
            elif isinstance(child, LogicalScan):
                return LogicalFilter(child=child, predicate=node.predicate)
            else:
                return LogicalFilter(child=child, predicate=node.predicate)
                
        if isinstance(node, LogicalProject):
            return LogicalProject(child=Optimizer.pushdown_predicates(node.child), exprs=node.exprs)
        elif isinstance(node, LogicalAggregate):
            return LogicalAggregate(child=Optimizer.pushdown_predicates(node.child), group_by=node.group_by, aggs=node.aggs)
        elif isinstance(node, LogicalSort):
            return LogicalSort(child=Optimizer.pushdown_predicates(node.child), order_by=node.order_by)
        elif isinstance(node, LogicalJoin):
            return LogicalJoin(
                left=Optimizer.pushdown_predicates(node.left),
                right=Optimizer.pushdown_predicates(node.right),
                on_expr=node.on_expr,
                join_type=node.join_type
            )
        elif isinstance(node, LogicalScan):
            return node
            
        return node

    @staticmethod
    def prune_columns(node: LogicalPlanNode) -> LogicalPlanNode:
        def _prune(node: LogicalPlanNode, required_cols: Set[str]) -> LogicalPlanNode:
            if isinstance(node, LogicalProject):
                req = set()
                for expr in node.exprs:
                    req.update(extract_columns(expr))
                return LogicalProject(child=_prune(node.child, req), exprs=node.exprs)
                
            elif isinstance(node, LogicalFilter):
                req = required_cols.copy()
                req.update(extract_columns(node.predicate))
                return LogicalFilter(child=_prune(node.child, req), predicate=node.predicate)
                
            elif isinstance(node, LogicalAggregate):
                req = required_cols.copy()
                for expr in node.group_by:
                    req.update(extract_columns(expr))
                for expr in node.aggs:
                    req.update(extract_columns(expr))
                return LogicalAggregate(child=_prune(node.child, req), group_by=node.group_by, aggs=node.aggs)
                
            elif isinstance(node, LogicalSort):
                req = required_cols.copy()
                for item in node.order_by:
                    req.update(extract_columns(item.expr))
                return LogicalSort(child=_prune(node.child, req), order_by=node.order_by)
                
            elif isinstance(node, LogicalJoin):
                req = required_cols.copy()
                req.update(extract_columns(node.on_expr))
                return LogicalJoin(
                    left=_prune(node.left, req),
                    right=_prune(node.right, req),
                    on_expr=node.on_expr,
                    join_type=node.join_type
                )
                
            elif isinstance(node, LogicalScan):
                cols = list(required_cols) if required_cols else None
                return LogicalScan(table_name=node.table_name, columns=cols)
                
            return node

        return _prune(node, set())
