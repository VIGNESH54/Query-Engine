import re
from typing import List, Tuple, cast
from .ast_nodes import Expr, Column, Literal, BinaryOp, JoinItem, SortItem, SelectStmt

class ParseError(Exception):
    pass

def tokenize(sql: str) -> List[Tuple[str, str]]:
    sql = sql.replace('\n', ' ').replace('\t', ' ')
    tokens = []
    # Simplified regex for tokens
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),
        ('STRING',   r"'[^']*'|\"[^\"]*\""),
        ('OP',       r'>=|<=|!=|=|>|<|\*|\+|\-|/'),
        ('ID',       r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?'),
        ('COMMA',    r','),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('WS',       r'\s+'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for mo in re.finditer(tok_regex, sql):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'WS':
            continue
        if kind == 'ID':
            val_upper = value.upper()
            if val_upper in {'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'ON', 'GROUP', 'BY', 'ORDER', 'ASC', 'DESC', 'AND', 'OR', 'AS'}:
                kind = val_upper
        tokens.append((kind, value))
    return tokens

class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', '')

    def consume(self, expected_kind: str = None) -> Tuple[str, str]:
        tok = self.peek()
        if expected_kind and tok[0] != expected_kind:
            raise ParseError(f"Expected {expected_kind}, got {tok[0]} at {tok[1]}")
        self.pos += 1
        return tok

    def match(self, kind: str) -> bool:
        if self.peek()[0] == kind:
            self.pos += 1
            return True
        return False

    def parse(self) -> SelectStmt:
        self.consume('SELECT')
        projections = self.parse_expr_list()
        
        self.consume('FROM')
        from_table = self.consume('ID')[1]
        
        joins = []
        while self.match('JOIN') or self.match('INNER'):
            if self.tokens[self.pos-1][0] == 'INNER':
                self.consume('JOIN')
            join_table = self.consume('ID')[1]
            self.consume('ON')
            on_expr = self.parse_expr()
            joins.append(JoinItem('INNER', join_table, on_expr))
            
        where_clause = None
        if self.match('WHERE'):
            where_clause = self.parse_expr()
            
        group_by = []
        if self.match('GROUP'):
            self.consume('BY')
            group_by = self.parse_expr_list()
            
        order_by = []
        if self.match('ORDER'):
            self.consume('BY')
            order_by = self.parse_order_list()
            
        return SelectStmt(projections, from_table, joins, where_clause, group_by, order_by)

    def parse_expr_list(self) -> List[Expr]:
        exprs = []
        if self.peek()[0] == 'OP' and self.peek()[1] == '*':
            self.consume('OP')
            return [] # Empty means *
        
        exprs.append(self.parse_expr())
        while self.match('COMMA'):
            exprs.append(self.parse_expr())
        return exprs

    def parse_order_list(self) -> List[SortItem]:
        items = []
        expr = self.parse_expr()
        desc = False
        if self.match('DESC'):
            desc = True
        elif self.match('ASC'):
            desc = False
        items.append(SortItem(expr, desc))
        
        while self.match('COMMA'):
            expr = self.parse_expr()
            desc = False
            if self.match('DESC'):
                desc = True
            elif self.match('ASC'):
                desc = False
            items.append(SortItem(expr, desc))
        return items

    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.match('OR'):
            right = self.parse_and()
            left = BinaryOp(left, 'OR', right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_comparison()
        while self.match('AND'):
            right = self.parse_comparison()
            left = BinaryOp(left, 'AND', right)
        return left

    def parse_comparison(self) -> Expr:
        left = self.parse_primary()
        if self.peek()[0] == 'OP':
            op = self.consume('OP')[1]
            right = self.parse_primary()
            return BinaryOp(left, op, right)
        return left

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if tok[0] == 'NUMBER':
            self.consume('NUMBER')
            if '.' in tok[1]:
                return Literal(float(tok[1]))
            return Literal(int(tok[1]))
        elif tok[0] == 'STRING':
            self.consume('STRING')
            return Literal(tok[1].strip("'\""))
        elif tok[0] == 'ID':
            self.consume('ID')
            val = tok[1]
            
            from .ast_nodes import AggFunc
            if val.upper() in ('SUM', 'COUNT', 'MIN', 'MAX', 'AVG') and self.peek()[0] == 'LPAREN':
                func_name = val.upper()
                self.consume('LPAREN')
                if func_name == 'COUNT' and self.peek()[0] == 'OP' and self.peek()[1] == '*':
                    self.consume('OP')  # consume *
                    inner_expr = Literal(1)
                else:
                    inner_expr = self.parse_expr()
                self.consume('RPAREN')
                return AggFunc(func_name, inner_expr)
                
            parts = val.split('.')
            if len(parts) == 2:
                return Column(parts[1], parts[0])
            return Column(parts[0])
        else:
            raise ParseError(f"Unexpected token in expression: {tok}")

def parse_sql(sql: str) -> SelectStmt:
    tokens = tokenize(sql)
    parser = Parser(tokens)
    return parser.parse()
