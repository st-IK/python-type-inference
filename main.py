"""
Python source analyzer / typed IR generator.

Usage:
    python main.py <input_python_file> <output_directory>

Outputs:
    <output_directory>/
        <input_stem>.tir
        <input_stem>.yaml

The .tir file is a human-readable typed intermediate representation.
The .yaml file is a machine-readable serialization of the same analysis.

This first version intentionally performs conservative static analysis:
- variable definitions and assignments
- functions and arguments
- classes, methods and class attributes
- type annotations
- basic literal/container/call-expression type inference
- Union types for variables assigned incompatible types
- Unknown when the type cannot be determined statically

It does not execute the analyzed Python program.
2026 - 05 - 05
"""

from __future__ import annotations

import ast
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


@dataclass(frozen=True)
class TypeInfo:
    kind: str
    name: str | None = None
    elements: tuple["TypeInfo", ...] = ()
    key: "TypeInfo | None" = None
    value: "TypeInfo | None" = None
    args: tuple["TypeInfo", ...] = ()
    returns: "TypeInfo | None" = None

    def sort_key(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        if self.kind == "unknown":
            return "Unknown"
        if self.kind == "any":
            return "Any"
        if self.kind == "primitive":
            return self.name or "Unknown"
        if self.kind == "named":
            return self.name or "Unknown"
        if self.kind == "list":
            return f"list[{self.elements[0].to_string()}]" if self.elements else "list[Any]"
        if self.kind == "set":
            return f"set[{self.elements[0].to_string()}]" if self.elements else "set[Any]"
        if self.kind == "tuple":
            if not self.elements:
                return "tuple[]"
            return "tuple[" + ", ".join(t.to_string() for t in self.elements) + "]"
        if self.kind == "dict":
            key = self.key.to_string() if self.key else "Any"
            value = self.value.to_string() if self.value else "Any"
            return f"dict[{key}, {value}]"
        if self.kind == "union":
            return " | ".join(t.to_string() for t in self.elements)
        if self.kind == "callable":
            args = ", ".join(t.to_string() for t in self.args)
            ret = self.returns.to_string() if self.returns else "Unknown"
            return f"Callable[[{args}], {ret}]"
        return self.kind


UNKNOWN = TypeInfo("unknown")
ANY = TypeInfo("any")


def primitive(name: str) -> TypeInfo:
    return TypeInfo("primitive", name=name)


def named(name: str) -> TypeInfo:
    return TypeInfo("named", name=name)


def union_types(types: list[TypeInfo]) -> TypeInfo:
    flattened: list[TypeInfo] = []
    for type_info in types:
        if type_info.kind == "union":
            flattened.extend(type_info.elements)
        else:
            flattened.append(type_info)

    unique: dict[str, TypeInfo] = {}
    for type_info in flattened:
        unique[type_info.to_string()] = type_info

    values = sorted(unique.values(), key=lambda x: x.sort_key())
    if not values:
        return UNKNOWN
    if len(values) == 1:
        return values[0]
    return TypeInfo("union", elements=tuple(values))


def merge_types(a: TypeInfo, b: TypeInfo) -> TypeInfo:
    if a == b:
        return a
    if a.kind == "unknown":
        return b
    if b.kind == "unknown":
        return a
    if a.kind == "any" or b.kind == "any":
        return ANY
    return union_types([a, b])


@dataclass
class AssignmentIR:
    line: int
    type: TypeInfo
    annotation: TypeInfo | None = None


@dataclass
class VariableIR:
    name: str
    declared_type: TypeInfo | None = None
    inferred_type: TypeInfo = UNKNOWN
    possible_types: list[TypeInfo] = field(default_factory=list)
    assignments: list[AssignmentIR] = field(default_factory=list)

    def add_assignment(
        self,
        line: int,
        inferred_type: TypeInfo,
        annotation: TypeInfo | None = None,
    ) -> None:
        if annotation is not None:
            self.declared_type = annotation

        self.inferred_type = merge_types(self.inferred_type, inferred_type)

        if not any(t.to_string() == inferred_type.to_string() for t in self.possible_types):
            self.possible_types.append(inferred_type)

        self.assignments.append(
            AssignmentIR(line=line, type=inferred_type, annotation=annotation)
        )


@dataclass
class ParameterIR:
    name: str
    type: TypeInfo = UNKNOWN
    default: str | None = None


@dataclass
class FunctionIR:
    name: str
    line: int
    arguments: list[ParameterIR] = field(default_factory=list)
    return_type: TypeInfo = UNKNOWN
    variables: dict[str, VariableIR] = field(default_factory=dict)
    returns: list[TypeInfo] = field(default_factory=list)
    is_method: bool = False


@dataclass
class ClassIR:
    name: str
    line: int
    bases: list[str] = field(default_factory=list)
    attributes: dict[str, VariableIR] = field(default_factory=dict)
    methods: dict[str, FunctionIR] = field(default_factory=dict)


@dataclass
class ModuleIR:
    source_file: str
    variables: dict[str, VariableIR] = field(default_factory=dict)
    functions: dict[str, FunctionIR] = field(default_factory=dict)
    classes: dict[str, ClassIR] = field(default_factory=dict)


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str, source_file: str) -> None:
        self.source = source
        self.module = ModuleIR(source_file=source_file)
        self.scope_stack: list[dict[str, VariableIR]] = [self.module.variables]
        self.function_stack: list[FunctionIR] = []
        self.class_stack: list[ClassIR] = []

    @property
    def current_scope(self) -> dict[str, VariableIR]:
        return self.scope_stack[-1]

    def current_function(self) -> FunctionIR | None:
        return self.function_stack[-1] if self.function_stack else None

    def current_class(self) -> ClassIR | None:
        return self.class_stack[-1] if self.class_stack else None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        function = FunctionIR(
            name=node.name,
            line=node.lineno,
            return_type=self.annotation_to_type(node.returns) or UNKNOWN,
            is_method=bool(self.class_stack),
        )

        positional_args = [*node.args.posonlyargs, *node.args.args]
        positional_default_start = len(positional_args) - len(node.args.defaults)

        for index, arg in enumerate(positional_args):
            default = None
            if index >= positional_default_start:
                default = self._unparse_default(
                    node.args.defaults[index - positional_default_start]
                )

            function.arguments.append(
                ParameterIR(
                    name=arg.arg,
                    type=self.annotation_to_type(arg.annotation) or UNKNOWN,
                    default=default,
                )
            )

        for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            function.arguments.append(
                ParameterIR(
                    name=arg.arg,
                    type=self.annotation_to_type(arg.annotation) or UNKNOWN,
                    default=self._unparse_default(default_node),
                )
            )

        if node.args.vararg:
            function.arguments.append(
                ParameterIR(
                    name="*" + node.args.vararg.arg,
                    type=self.annotation_to_type(node.args.vararg.annotation) or UNKNOWN,
                )
            )

        if node.args.kwarg:
            function.arguments.append(
                ParameterIR(
                    name="**" + node.args.kwarg.arg,
                    type=self.annotation_to_type(node.args.kwarg.annotation) or UNKNOWN,
                )
            )

        if self.current_class() is not None:
            self.current_class().methods[node.name] = function
        else:
            self.module.functions[node.name] = function

        self.function_stack.append(function)
        self.scope_stack.append(function.variables)

        for parameter in function.arguments:
            clean_name = parameter.name.lstrip("*")
            if clean_name:
                variable = function.variables.setdefault(
                    clean_name, VariableIR(name=clean_name)
                )
                variable.declared_type = parameter.type
                variable.inferred_type = parameter.type

        for statement in node.body:
            self.visit(statement)

        if function.returns:
            function.return_type = union_types(function.returns)

        self.scope_stack.pop()
        self.function_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        class_ir = ClassIR(
            name=node.name,
            line=node.lineno,
            bases=[self.expr_to_name(base) for base in node.bases],
        )
        self.module.classes[node.name] = class_ir

        self.class_stack.append(class_ir)
        self.scope_stack.append(class_ir.attributes)

        for statement in node.body:
            self.visit(statement)

        self.scope_stack.pop()
        self.class_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        inferred = self.infer_expr(node.value)
        for target in node.targets:
            self.record_target(target, inferred, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        annotation = self.annotation_to_type(node.annotation)
        inferred = self.infer_expr(node.value) if node.value else annotation or UNKNOWN
        final_type = annotation if node.value is None else merge_types(annotation or UNKNOWN, inferred)

        self.record_target(node.target, final_type, node.lineno, annotation=annotation)

        if node.value:
            self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Any:
        inferred = self.infer_expr(node.value)
        self.record_target(node.target, inferred, node.lineno)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        iterable_type = self.infer_expr(node.iter)
        self.record_target(node.target, self.iterable_element_type(iterable_type), node.lineno)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        iterable_type = self.infer_expr(node.iter)
        self.record_target(node.target, self.iterable_element_type(iterable_type), node.lineno)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> Any:
        for item in node.items:
            if item.optional_vars is not None:
                self.record_target(
                    item.optional_vars,
                    self.infer_expr(item.context_expr),
                    node.lineno,
                )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        self.visit_With(node)

    def visit_Return(self, node: ast.Return) -> Any:
        if self.current_function() is not None:
            self.current_function().returns.append(
                self.infer_expr(node.value) if node.value else primitive("None")
            )
        self.generic_visit(node)

    def record_target(
        self,
        target: ast.AST,
        type_info: TypeInfo,
        line: int,
        annotation: TypeInfo | None = None,
    ) -> None:
        if isinstance(target, ast.Name):
            variable = self.current_scope.setdefault(target.id, VariableIR(name=target.id))
            variable.add_assignment(line, type_info, annotation)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.record_target(element, UNKNOWN, line)

    def infer_expr(self, node: ast.AST | None) -> TypeInfo:
        if node is None:
            return UNKNOWN

        if isinstance(node, ast.Constant):
            value = node.value
            if value is None:
                return primitive("None")
            if isinstance(value, bool):
                return primitive("bool")
            if isinstance(value, int):
                return primitive("int")
            if isinstance(value, float):
                return primitive("float")
            if isinstance(value, complex):
                return primitive("complex")
            if isinstance(value, str):
                return primitive("str")
            if isinstance(value, bytes):
                return primitive("bytes")
            return UNKNOWN

        if isinstance(node, ast.List):
            return TypeInfo("list", elements=(self.infer_element_union(node.elts),))

        if isinstance(node, ast.Set):
            return TypeInfo("set", elements=(self.infer_element_union(node.elts),))

        if isinstance(node, ast.Tuple):
            return TypeInfo("tuple", elements=tuple(self.infer_expr(x) for x in node.elts))

        if isinstance(node, ast.Dict):
            key_types = [self.infer_expr(key) for key in node.keys if key is not None]
            value_types = [self.infer_expr(value) for value in node.values]
            return TypeInfo(
                "dict",
                key=union_types(key_types) if key_types else ANY,
                value=union_types(value_types) if value_types else ANY,
            )

        if isinstance(node, ast.ListComp):
            return TypeInfo("list", elements=(self.infer_expr(node.elt),))

        if isinstance(node, ast.SetComp):
            return TypeInfo("set", elements=(self.infer_expr(node.elt),))

        if isinstance(node, ast.DictComp):
            return TypeInfo("dict", key=self.infer_expr(node.key), value=self.infer_expr(node.value))

        if isinstance(node, ast.GeneratorExp):
            return named("Generator")

        if isinstance(node, ast.Lambda):
            return TypeInfo("callable", returns=UNKNOWN)

        if isinstance(node, ast.JoinedStr):
            return primitive("str")

        if isinstance(node, ast.BinOp):
            return self.infer_binop(self.infer_expr(node.left), self.infer_expr(node.right), node.op)

        if isinstance(node, ast.UnaryOp):
            operand = self.infer_expr(node.operand)
            return primitive("bool") if isinstance(node.op, ast.Not) else operand

        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return primitive("bool")

        if isinstance(node, ast.IfExp):
            return union_types([self.infer_expr(node.body), self.infer_expr(node.orelse)])

        if isinstance(node, ast.Call):
            return self.infer_call(node)

        if isinstance(node, ast.Name):
            variable = self.lookup_variable(node.id)
            if variable is not None:
                return variable.inferred_type

            builtins = {
                "int": primitive("int"),
                "float": primitive("float"),
                "str": primitive("str"),
                "bool": primitive("bool"),
                "bytes": primitive("bytes"),
                "list": named("list"),
                "dict": named("dict"),
                "set": named("set"),
                "tuple": named("tuple"),
                "complex": primitive("complex"),
            }
            return builtins.get(node.id, UNKNOWN)

        if isinstance(node, ast.Subscript):
            base = self.infer_expr(node.value)
            if base.kind == "list" and base.elements:
                return base.elements[0]
            if base.kind == "tuple" and base.elements:
                return union_types(list(base.elements))
            if base.kind == "dict" and base.value:
                return base.value
            return UNKNOWN

        if isinstance(node, ast.Await):
            return self.infer_expr(node.value)

        return UNKNOWN

    def infer_element_union(self, elements: list[ast.AST]) -> TypeInfo:
        return ANY if not elements else union_types([self.infer_expr(element) for element in elements])

    def infer_binop(self, left: TypeInfo, right: TypeInfo, op: ast.operator) -> TypeInfo:
        numeric = {"int", "float", "complex"}
        if left.kind == "primitive" and right.kind == "primitive":
            if left.name == "str" and right.name == "str" and isinstance(op, ast.Add):
                return primitive("str")
            if left.name in numeric and right.name in numeric:
                if "complex" in {left.name, right.name}:
                    return primitive("complex")
                if "float" in {left.name, right.name}:
                    return primitive("float")
                return primitive("int")
        return UNKNOWN

    def infer_call(self, node: ast.Call) -> TypeInfo:
        if isinstance(node.func, ast.Name):
            builtin_returns = {
                "int": primitive("int"),
                "float": primitive("float"),
                "str": primitive("str"),
                "bool": primitive("bool"),
                "bytes": primitive("bytes"),
                "list": named("list"),
                "dict": named("dict"),
                "set": named("set"),
                "tuple": named("tuple"),
            }
            if node.func.id in builtin_returns:
                return builtin_returns[node.func.id]

            function = self.module.functions.get(node.func.id)
            if function is not None:
                return function.return_type

        return UNKNOWN

    def iterable_element_type(self, type_info: TypeInfo) -> TypeInfo:
        if type_info.kind in {"list", "set"} and type_info.elements:
            return type_info.elements[0]
        if type_info.kind == "tuple":
            return union_types(list(type_info.elements))
        if type_info.kind == "dict":
            return type_info.key or UNKNOWN
        return UNKNOWN

    def lookup_variable(self, name: str) -> VariableIR | None:
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None

    def annotation_to_type(self, node: ast.AST | None) -> TypeInfo | None:
        if node is None:
            return None

        if isinstance(node, ast.Name):
            builtin_map = {
                "int": primitive("int"),
                "float": primitive("float"),
                "str": primitive("str"),
                "bool": primitive("bool"),
                "bytes": primitive("bytes"),
                "None": primitive("None"),
                "Any": ANY,
            }
            return builtin_map.get(node.id, named(node.id))

        if isinstance(node, ast.Constant) and node.value is None:
            return primitive("None")

        if isinstance(node, ast.Attribute):
            return named(f"{self.expr_to_name(node.value)}.{node.attr}")

        if isinstance(node, ast.Subscript):
            base = self.expr_to_name(node.value)
            args = self.subscript_args(node.slice)
            base_lower = base.lower()

            if base_lower in {"list", "set"}:
                element = union_types(args) if args else ANY
                return TypeInfo("list" if base_lower == "list" else "set", elements=(element,))
            if base_lower == "tuple":
                return TypeInfo("tuple", elements=tuple(args))
            if base_lower == "dict":
                return TypeInfo(
                    "dict",
                    key=args[0] if len(args) > 0 else ANY,
                    value=args[1] if len(args) > 1 else ANY,
                )
            if base_lower in {"union", "typing.union"}:
                return union_types(args)
            if base_lower == "optional":
                return union_types([*args, primitive("None")])

            return named(f"{base}[{', '.join(t.to_string() for t in args)}]")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return union_types([
                self.annotation_to_type(node.left) or UNKNOWN,
                self.annotation_to_type(node.right) or UNKNOWN,
            ])

        return named(self._safe_unparse(node))

    def subscript_args(self, node: ast.AST) -> list[TypeInfo]:
        if isinstance(node, ast.Tuple):
            return [self.annotation_to_type(element) or UNKNOWN for element in node.elts]
        return [self.annotation_to_type(node) or UNKNOWN]

    def expr_to_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self.expr_to_name(node.value)}.{node.attr}"
        return self._safe_unparse(node)

    def _safe_unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expression>"

    def _unparse_default(self, node: ast.AST | None) -> str | None:
        return None if node is None else self._safe_unparse(node)


def type_to_dict(type_info: TypeInfo) -> dict[str, Any] | str:
    if type_info.kind in {"unknown", "any", "primitive", "named"}:
        return type_info.to_string()

    if type_info.kind in {"list", "set", "tuple", "union"}:
        return {
            "kind": type_info.kind,
            "types": [type_to_dict(t) for t in type_info.elements],
        }

    if type_info.kind == "dict":
        return {
            "kind": "dict",
            "key": type_to_dict(type_info.key or ANY),
            "value": type_to_dict(type_info.value or ANY),
        }

    if type_info.kind == "callable":
        return {
            "kind": "callable",
            "args": [type_to_dict(t) for t in type_info.args],
            "returns": type_to_dict(type_info.returns or UNKNOWN),
        }

    return type_info.to_string()


def variable_to_dict(variable: VariableIR) -> dict[str, Any]:
    result: dict[str, Any] = {
        "inferred_type": type_to_dict(variable.inferred_type),
        "possible_types": [
            type_to_dict(t)
            for t in sorted(variable.possible_types, key=lambda x: x.sort_key())
        ],
        "assignments": [],
    }

    if variable.declared_type is not None:
        result["declared_type"] = type_to_dict(variable.declared_type)

    for assignment in variable.assignments:
        item: dict[str, Any] = {
            "line": assignment.line,
            "type": type_to_dict(assignment.type),
        }
        if assignment.annotation is not None:
            item["annotation"] = type_to_dict(assignment.annotation)
        result["assignments"].append(item)

    return result


def function_to_dict(function: FunctionIR) -> dict[str, Any]:
    return {
        "line": function.line,
        "arguments": {
            argument.name: {
                "type": type_to_dict(argument.type),
                **({"default": argument.default} if argument.default is not None else {}),
            }
            for argument in function.arguments
        },
        "return_type": type_to_dict(function.return_type),
        "variables": {
            name: variable_to_dict(variable)
            for name, variable in function.variables.items()
        },
    }


def class_to_dict(class_ir: ClassIR) -> dict[str, Any]:
    return {
        "line": class_ir.line,
        "bases": class_ir.bases,
        "attributes": {
            name: variable_to_dict(variable)
            for name, variable in class_ir.attributes.items()
        },
        "methods": {
            name: function_to_dict(function)
            for name, function in class_ir.methods.items()
        },
    }


def module_to_dict(module: ModuleIR) -> dict[str, Any]:
    return {
        "format": "python-typed-ir",
        "version": "0.1",
        "source_file": module.source_file,
        "variables": {
            name: variable_to_dict(variable)
            for name, variable in module.variables.items()
        },
        "functions": {
            name: function_to_dict(function)
            for name, function in module.functions.items()
        },
        "classes": {
            name: class_to_dict(class_ir)
            for name, class_ir in module.classes.items()
        },
    }


def build_tir(module: ModuleIR) -> str:
    lines: list[str] = ["typed-ir version=0.1", f"source {module.source_file}", ""]

    if module.variables:
        lines.append("variables:")
        for name, variable in module.variables.items():
            lines.append(f"  {name}: {variable.inferred_type.to_string()}")
            if variable.declared_type is not None:
                lines.append(f"    declared: {variable.declared_type.to_string()}")
            if variable.possible_types:
                types = ", ".join(
                    t.to_string() for t in sorted(variable.possible_types, key=lambda x: x.sort_key())
                )
                lines.append(f"    possible: [{types}]")
            for assignment in variable.assignments:
                lines.append(
                    f"    assignment line={assignment.line}: {assignment.type.to_string()}"
                )
        lines.append("")

    if module.functions:
        lines.append("functions:")
        for name, function in module.functions.items():
            lines.append(f"  {name}:")
            lines.append(f"    line: {function.line}")
            lines.append(f"    return: {function.return_type.to_string()}")
            lines.append("    arguments:")
            for argument in function.arguments:
                suffix = f" = {argument.default}" if argument.default is not None else ""
                lines.append(f"      {argument.name}: {argument.type.to_string()}{suffix}")
            if function.variables:
                lines.append("    variables:")
                for variable in function.variables.values():
                    lines.append(f"      {variable.name}: {variable.inferred_type.to_string()}")
            lines.append("")

    if module.classes:
        lines.append("classes:")
        for name, class_ir in module.classes.items():
            lines.append(f"  {name}:")
            lines.append(f"    line: {class_ir.line}")
            if class_ir.bases:
                lines.append(f"    bases: {', '.join(class_ir.bases)}")
            if class_ir.attributes:
                lines.append("    attributes:")
                for attribute in class_ir.attributes.values():
                    lines.append(f"      {attribute.name}: {attribute.inferred_type.to_string()}")
            if class_ir.methods:
                lines.append("    methods:")
                for method in class_ir.methods.values():
                    lines.append(f"      {method.name}:")
                    lines.append(f"        line: {method.line}")
                    lines.append(f"        return: {method.return_type.to_string()}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def analyze_file(input_path: Path) -> ModuleIR:
    source = input_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(input_path))
    except SyntaxError as exc:
        raise SystemExit(
            f"Syntax error in {input_path}:{exc.lineno}:{exc.offset}: {exc.msg}"
        ) from exc

    analyzer = PythonAnalyzer(source, input_path.name)
    analyzer.visit(tree)
    return analyzer.module


def save_outputs(module: ModuleIR, input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    tir_path = output_dir / f"{stem}.tir"
    yaml_path = output_dir / f"{stem}.yaml"

    tir_path.write_text(build_tir(module), encoding="utf-8")
    yaml_path.write_text(
        yaml.safe_dump(
            module_to_dict(module),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )

    print(f"[Parser] Input : {input_path}")
    print(f"[Parser] Output: {output_dir}")
    print(f"[Parser] IR    : {tir_path}")
    print(f"[Parser] YAML  : {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Parser",
        description="Analyze one Python source file and generate typed IR + YAML.",
    )
    parser.add_argument("input_python_file", help="Python source file to analyze.")
    parser.add_argument("output_directory", help="Directory for generated files.")
    args = parser.parse_args()

    input_path = Path(args.input_python_file).resolve()
    output_dir = Path(args.output_directory).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() != ".py":
        raise SystemExit(f"Input file must be a .py file: {input_path}")

    module = analyze_file(input_path)
    save_outputs(module, input_path, output_dir)


if __name__ == "__main__":
    main()
