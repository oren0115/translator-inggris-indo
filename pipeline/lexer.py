"""Lexer: memecah teks menjadi aliran token (kata, angka, tanda baca)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


class TokenType(Enum):
    """Kategori token seperti pada lexer bahasa pemrograman."""

    WORD = auto()
    NUMBER = auto()
    PUNCT = auto()
    WHITESPACE = auto()
    NEWLINE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:C{self.column})"


class Lexer:
    """
    Scanner berbasis regex yang menghasilkan token stream.
    Mirip lexer compiler: satu pass, posisi baris/kolom untuk error reporting.
    """

    _pattern = re.compile(
        r"""
        (?P<WORD>[\w'-]+) |
        (?P<NUMBER>\d+(?:[.,]\d+)*) |
        (?P<NEWLINE>\n) |
        (?P<WHITESPACE>[ \t\r]+) |
        (?P<PUNCT>[^\w\s])
        """,
        re.VERBOSE,
    )

    def __init__(self, source: str) -> None:
        self._source = source

    def tokenize(self) -> list[Token]:
        return list(self.iter_tokens())

    def iter_tokens(self) -> Iterator[Token]:
        line = 1
        col = 1
        pos = 0
        n = len(self._source)

        while pos < n:
            m = self._pattern.match(self._source, pos)
            if not m:
                yield Token(TokenType.UNKNOWN, self._source[pos], line, col)
                pos += 1
                col += 1
                continue

            kind = m.lastgroup
            assert kind is not None
            raw = m.group(0)
            pos = m.end()

            if kind == "NEWLINE":
                yield Token(TokenType.NEWLINE, raw, line, col)
                line += 1
                col = 1
            elif kind == "WHITESPACE":
                yield Token(TokenType.WHITESPACE, raw, line, col)
                col += len(raw)
            else:
                t = TokenType[kind]
                yield Token(t, raw, line, col)
                col += len(raw)

    @staticmethod
    def significant_tokens(tokens: list[Token]) -> list[Token]:
        """Token yang dipakai parser (tanpa whitespace)."""

        skip = {TokenType.WHITESPACE}
        return [t for t in tokens if t.type not in skip]
