from dataclasses import dataclass
from typing import List, Optional, Union, Any

@dataclass
class Expr:
    pass

@dataclass
class Column(Expr):
    name: str
    table: Optional[str] = None
    
    def __hash__(self):
        return hash((self.name, self.table))

@dataclass
class Literal(Expr):
    value: Any

@dataclass
class AggFunc(Expr):
    func: str
    expr: Expr
    alias: Optional[str] = None

@dataclass
class BinaryOp(Expr):
    left: Expr
    op: str
    right: Expr

@dataclass
class JoinItem:
    join_type: str  # e.g., 'INNER'
    table: str
    on_expr: Expr

@dataclass
class SortItem:
    expr: Expr
    desc: bool = False

@dataclass
class SelectStmt:
    projections: List[Expr]  # if empty, implies *
    from_table: str
    joins: List[JoinItem]
    where_clause: Optional[Expr]
    group_by: List[Expr]
    order_by: List[SortItem]
