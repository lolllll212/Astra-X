"""Safe arithmetic expression evaluator.

Uses Python's ``ast`` module to parse and evaluate mathematical
expressions.  Only literals, operators, and a whitelist of math
functions are allowed.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_CONSTS: frozenset[str] = frozenset({"pi", "e"})


class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression safely. Supports +, -, *, /, //, %, **, and functions like sqrt, sin, cos, log, pi, e."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="expression", type_="string", description="Mathematical expression to evaluate", required=True),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        expression: str = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="expression is required")
        try:
            tree: ast.Expression = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            return ToolResult(success=False, error=f"Syntax error: {exc}")

        if not isinstance(tree, ast.Expression):
            return ToolResult(success=False, error="Not a valid expression")

        try:
            value = self._eval_node(tree.body)
        except (ValueError, TypeError, ArithmeticError) as exc:
            return ToolResult(success=False, error=str(exc))

        output = str(value)
        return ToolResult(success=True, output=output)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant: {type(node.value).__name__}")
            return node.value

        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.operand))

        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ""
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"Unsupported function: '{func_name}'")
            args = [self._eval_node(a) for a in node.args]
            return _ALLOWED_FUNCS[func_name](*args)

        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return getattr(math, node.id)
            raise ValueError(f"Unknown identifier: '{node.id}'")

        raise ValueError(f"Unsupported expression: {type(node).__name__}")
