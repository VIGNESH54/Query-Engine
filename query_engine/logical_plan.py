from dataclasses import dataclass, field
from typing import List, Optional, Any, Set
from .ast_nodes import Expr, Column, SelectStmt

@dataclass
class LogicalPlanNode:
    def children(self) -> List['LogicalPlanNode']:
        return []

@dataclass
class LogicalScan(LogicalPlanNode):
    table_name: str
    columns: Optional[List[str]] = None  # None means all columns

@dataclass
class LogicalFilter(LogicalPlanNode):
    child: LogicalPlanNode
    predicate: Expr
    
    def children(self) -> List[LogicalPlanNode]:
        return [self.child]

@dataclass
class LogicalProject(LogicalPlanNode):
    child: LogicalPlanNode
    exprs: List[Expr]
    
    def children(self) -> List[LogicalPlanNode]:
        return [self.child]

@dataclass
class LogicalJoin(LogicalPlanNode):
    left: LogicalPlanNode
    right: LogicalPlanNode
    on_expr: Expr
    join_type: str = 'INNER'
    
    def children(self) -> List[LogicalPlanNode]:
        return [self.left, self.right]

@dataclass
class LogicalAggregate(LogicalPlanNode):
    child: LogicalPlanNode
    group_by: List[Expr]
    aggs: List[Expr] = field(default_factory=list)
    
    def children(self) -> List[LogicalPlanNode]:
        return [self.child]

@dataclass
class LogicalSort(LogicalPlanNode):
    child: LogicalPlanNode
    order_by: List[Any]  # List[SortItem] from AST
    
    def children(self) -> List[LogicalPlanNode]:
        return [self.child]

def build_logical_plan(ast: SelectStmt) -> LogicalPlanNode:
    # 1. FROM clause -> Scan
    plan = LogicalScan(table_name=ast.from_table)
    
    # 2. JOINs
    for join in ast.joins:
        right_scan = LogicalScan(table_name=join.table)
        plan = LogicalJoin(
            left=plan,
            right=right_scan,
            on_expr=join.on_expr,
            join_type=join.join_type
        )
        
    # 3. WHERE clause -> Filter
    if ast.where_clause:
        plan = LogicalFilter(child=plan, predicate=ast.where_clause)
        
    # 4. GROUP BY -> Aggregate (also extract AggFuncs from projections)
    from .ast_nodes import AggFunc
    agg_funcs = []
    
    def extract_aggs(expr):
        if isinstance(expr, AggFunc):
            agg_funcs.append(expr)
        elif hasattr(expr, 'left'):
            extract_aggs(expr.left)
            extract_aggs(expr.right)
            
    for proj in ast.projections:
        extract_aggs(proj)
        
    if ast.group_by or agg_funcs:
        plan = LogicalAggregate(child=plan, group_by=ast.group_by, aggs=agg_funcs)

    # 5. SELECT -> Project
    if ast.projections:
        plan = LogicalProject(child=plan, exprs=ast.projections)
        
    # 6. ORDER BY -> Sort
    if ast.order_by:
        plan = LogicalSort(child=plan, order_by=ast.order_by)
        
    return plan
