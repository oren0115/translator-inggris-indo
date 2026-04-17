"""Parser: membangun parse tree sederhana dengan frasa NP dan VP."""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Token, TokenType


_VERBS_EN = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "go", "goes", "went", "come", "comes", "came",
    "make", "makes", "made", "get", "gets", "got", "take", "takes", "took",
    "see", "sees", "saw", "know", "knows", "knew", "think", "thinks", "thought",
    "say", "says", "said", "tell", "tells", "told", "ask", "asks", "asked",
    "want", "wants", "wanted", "need", "needs", "needed", "like", "likes", "liked",
    "work", "works", "worked", "play", "plays", "played", "read", "reads",
    "write", "writes", "wrote", "eat", "eats", "ate", "drink", "drinks", "drank",
    "run", "runs", "ran", "walk", "walks", "walked", "live", "lives", "lived",
    "love", "loves", "loved", "help", "helps", "helped", "learn", "learns", "learned",
    "sit", "sits", "sat", "stand", "stands", "stood", "give", "gives", "gave",
    "find", "finds", "found", "use", "uses", "used", "try", "tries", "tried",
}
_VERBS_ID = {
    "adalah", "ialah", "merupakan", "ada", "adanya", "punya", "memiliki", "mempunyai",
    "pergi", "pergilah", "datang", "membuat", "membaca", "menulis", "makan", "minum",
    "berlari", "berjalan", "hidup", "tinggal", "suka", "menyukai", "membantu",
    "belajar", "mempelajari", "melihat", "mendengar", "berkata", "berbicara",
    "bertanya", "ingin", "butuh", "membutuhkan", "akan", "bisa", "dapat", "harus",
    "sudah", "telah", "sedang", "belum", "tidak", "bukan", "jangan", "duduk", "duduklah",
    "berdiri", "memberi", "mencoba", "menggunakan", "menemukan",
}
_VERBS = _VERBS_EN | _VERBS_ID


def _is_verb_lemma(word: str) -> bool:
    w = word.lower().strip("'")
    if w in _VERBS:
        return True
    if len(w) > 4 and w.endswith("ing") and w[:-3] in _VERBS_EN:
        return True
    if w.endswith("kan") and len(w) > 4:
        return True
    return False


@dataclass
class ParseNode:
    """Simpul pohon sintaks: S, NP, VP, atau daun TOKEN."""

    label: str
    children: list["ParseNode"] = field(default_factory=list)
    token: Token | None = None

    def leaves_text(self) -> str:
        if self.token is not None:
            return self.token.value
        return " ".join(c.leaves_text() for c in self.children if c.leaves_text())


class Parser:
    """
    Parser heuristik per klausa: NP = kata sebelum kata kerja pertama; VP = kata kerja + sisanya
    sampai tanda baca akhir kalimat.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = [t for t in tokens if t.type != TokenType.WHITESPACE]

    def parse(self) -> ParseNode:
        root = ParseNode("S")
        if not self._tokens:
            return root

        buf: list[Token] = []

        def flush_clause(buf_tokens: list[Token]) -> None:
            if not buf_tokens:
                return
            punct_only = all(
                t.type in (TokenType.PUNCT, TokenType.NEWLINE) for t in buf_tokens
            )
            if punct_only:
                for t in buf_tokens:
                    lab = "NEWLINE" if t.type == TokenType.NEWLINE else "PUNCT"
                    root.children.append(ParseNode(lab, token=t))
                return

            clause_parts: list[ParseNode] = []
            np = ParseNode("NP")
            vp = ParseNode("VP")
            seen_verb = False

            def flush_np() -> None:
                nonlocal np
                if np.children:
                    clause_parts.append(np)
                    np = ParseNode("NP")

            def flush_vp() -> None:
                nonlocal vp
                if vp.children:
                    clause_parts.append(vp)
                    vp = ParseNode("VP")

            for t in buf_tokens:
                if t.type == TokenType.PUNCT:
                    flush_np()
                    flush_vp()
                    clause_parts.append(ParseNode("PUNCT", token=t))
                    continue
                if t.type == TokenType.NEWLINE:
                    flush_np()
                    flush_vp()
                    clause_parts.append(ParseNode("NEWLINE", token=t))
                    continue
                if t.type == TokenType.WORD:
                    if not seen_verb and _is_verb_lemma(t.value):
                        flush_np()
                        seen_verb = True
                        vp.children.append(ParseNode("TOKEN", token=t))
                    elif not seen_verb:
                        np.children.append(ParseNode("TOKEN", token=t))
                    else:
                        vp.children.append(ParseNode("TOKEN", token=t))
                else:
                    flush_np()
                    flush_vp()
                    clause_parts.append(ParseNode("LEX", token=t))

            flush_np()
            flush_vp()
            root.children.extend(clause_parts)

        for t in self._tokens:
            if t.type == TokenType.NEWLINE:
                flush_clause(buf)
                buf = []
                root.children.append(ParseNode("NEWLINE", token=t))
                continue
            if t.type == TokenType.PUNCT and t.value in ".?!":
                buf.append(t)
                flush_clause(buf)
                buf = []
            else:
                buf.append(t)
        flush_clause(buf)
        return root

    @staticmethod
    def tree_to_str(node: ParseNode, indent: int = 0) -> str:
        pad = "  " * indent
        if node.token is not None:
            return f"{pad}{node.label}: {node.token.value!r} ({node.token.type.name})\n"
        lines = f"{pad}{node.label}\n"
        for c in node.children:
            lines += Parser.tree_to_str(c, indent + 1)
        return lines
