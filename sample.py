from typing import Optional

x = 100
x = "text"

ex_list = [1, "text", [2, "word"]]

name: str = "Alice"
values: list[int] = [1, 2, 3]


def add(a: int, b: float) -> float:
    result = a + b
    return result


def mixed(condition: bool):
    value = 10
    if condition:
        value = "hello"
    return value


class Example:
    count: int = 0

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name
