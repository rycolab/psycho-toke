import html
from IPython.display import HTML, display

from arsenal import colors


class Chart(dict):
    def __init__(self, zero, vals=()):
        self.zero = zero
        super().__init__(vals)

    def __missing__(self, k):
        return self.zero

    def spawn(self):
        return Chart(self.zero)

    def __add__(self, other):
        new = self.spawn()
        for k, v in self.items():
            new[k] += v
        for k, v in other.items():
            new[k] += v
        return new

    def __mul__(self, other):
        new = self.spawn()
        for k in self:
            v = self[k] * other[k]
            if v == self.zero:
                continue
            new[k] += v
        return new

    def copy(self):
        return Chart(self.zero, self)

    def trim(self):
        return Chart(
            self.zero, {k: v for k, v in self.items() if v != self.zero}
        )

    def metric(self, other):
        assert isinstance(other, Chart)
        err = 0
        for x in self.keys() | other.keys():
            err = max(err, abs(self[x] - other[x]))
        return err

    def _repr_html_(self):
        return (
            '<div style="font-family: Monospace;">'
            + format_table(self.trim().items(), headings=['key', 'value'])
            + '</div>'
        )

    def __repr__(self):
        return repr({k: v for k, v in self.items() if v != self.zero})

    def __str__(self, style_value=lambda k, v: str(v)):
        def key(k):
            return -self[k]

        return (
            'Chart {\n'
            + '\n'.join(
                f'  {k!r}: {style_value(k, self[k])},'
                for k in sorted(self, key=key)
                if self[k] != self.zero
            )
            + '\n}'
        )

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        if not isinstance(want, Chart):
            want = Chart(self.zero, want)
        if domain is None:
            domain = self.keys() | want.keys()
        assert verbose or throw
        errors = []
        for x in domain:
            if abs(self[x] - want[x]) <= tol:
                if verbose:
                    print(colors.mark(True), x, self[x])
            else:
                if verbose:
                    print(colors.mark(False), x, self[x], want[x])
                errors.append(x)
        if throw:
            for x in errors:
                raise AssertionError(f'{x}: {self[x]} {want[x]}')

    def argmax(self):
        return max(self, key=self.__getitem__)

    def argmin(self):
        return min(self, key=self.__getitem__)

    def top(self, k):
        return Chart(
            self.zero,
            {k: self[k] for k in sorted(self, key=self.__getitem__, reverse=True)[:k]},
        )

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())

    def sum(self):
        return sum(self.values())

    def sort(self, **kwargs):
        return Chart(self.zero, [(k, self[k]) for k in sorted(self, **kwargs)])

    def sort_descending(self):
        return Chart(self.zero, [(k, self[k]) for k in sorted(self, key=lambda k: -self[k])])

    def normalize(self):
        Z = self.sum()
        if Z == 0:
            return self
        return Chart(self.zero, [(k, v / Z) for k, v in self.items()])

    def filter(self, f):
        return Chart(self.zero, [(k, v) for k, v in self.items() if f(k)])

    def project(self, f):
        "Apply the function `f` to each key; summing when f-transformed keys overlap."
        out = self.spawn()
        for k, v in self.items():
            out[f(k)] += v
        return out

    # TODO: the more general version of this method is join
    def compare(self, other, *, domain=None):
        import pandas as pd

        if not isinstance(other, Chart):
            other = Chart(self.zero, other)
        if domain is None:
            domain = self.keys() | other.keys()
        rows = []
        for x in domain:
            m = abs(self[x] - other[x])
            rows.append(dict(key=x, self=self[x], other=other[x], metric=m))
        return pd.DataFrame(rows)


def prefixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(prefixes(''))
      ['']

      >>> list(prefixes('abc'))
      ['', 'a', 'ab', 'abc']

    """
    for p in range(len(z) + 1):
        yield z[:p]


class max_munch:
    def __init__(self, tokens):
        self._end = object()
        self.root = self.make_trie(tokens)

    def __call__(self, x):
        if len(x) == 0:
            return ()
        else:
            t, ys = self.munch(x)
            return (ys,) + self(x[t:])

    def munch(self, x):
        (t, ys) = next(self.traverse(x, 0, self.root))
        return (t, ys)

    def make_trie(self, words):
        root = dict()
        for word in words:
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def traverse(self, query, t, node):
        """
        Enumerate (in order of longest to shortest) the strings in the trie matching
        prefixes of `query`.
        """
        if node == self._end:
            return
        if t < len(query):
            x = query[t]
            if x in node:
                yield from self.traverse(query, t + 1, node[x])
        if self._end in node:
            yield (t, query[:t])  # post order gives the longest match


def color_code_alignment(seq1, seq2):
    import Levenshtein as lev
    from rich.console import Console
    from rich.text import Text

    alignment = lev.editops(seq1, seq2)
    console = Console()

    colored_seq1 = Text()
    colored_seq2 = Text()

    seq1 = [f'{x}|' for x in seq1]
    seq2 = [f'{x}|' for x in seq2]

    idx1, idx2 = 0, 0
    for op, i, j in alignment:
        while idx1 < i:
            colored_seq1.append(seq1[idx1], style="green")
            idx1 += 1
        while idx2 < j:
            colored_seq2.append(seq2[idx2], style="green")
            idx2 += 1

        if op == 'replace':
            colored_seq1.append(seq1[idx1], style="red")
            colored_seq2.append(seq2[idx2], style="red")
            idx1 += 1
            idx2 += 1
        elif op == 'insert':
            colored_seq2.append(seq2[idx2], style="blue")
            idx2 += 1
        elif op == 'delete':
            colored_seq1.append(seq1[idx1], style="yellow")
            idx1 += 1

    while idx1 < len(seq1):
        colored_seq1.append(seq1[idx1], style="green")
        idx1 += 1
    while idx2 < len(seq2):
        colored_seq2.append(seq2[idx2], style="green")
        idx2 += 1

    console.print("Sequence 1:")
    console.print(colored_seq1)
    console.print("Sequence 2:")
    console.print(colored_seq2)


def flatten(xs):
    if len(xs) == 0:
        return ()
    else:
        ys, y = xs
        return flatten(ys) + (y,)


def unflatten(ys):
    xs = ()
    for y in ys:
        xs = (xs, y)
    return xs


def longest_common_prefix(xs):
    if not xs:
        return ""

    # Sort the strings
    xs = sorted(xs)

    # Compare only the first and the last strings
    first = xs[0]
    last = xs[-1]

    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1

    # The longest common prefix will be the portion of the first string up to i
    return first[:i]


def prefix(xs, ys):
    assert isinstance(xs, str) and isinstance(ys, str)
    return ys.startswith(xs)


def strict_prefix(xs, ys):
    assert isinstance(xs, str) and isinstance(ys, str)
    return prefix(xs, ys) and xs != ys


def cons2str(ys):
    xs = []
    while ys != ():
        ys, y = ys
        xs.append(y)
    return ''.join(reversed(xs))


def covers(qs, ys):
    assert isinstance(qs, str) and isinstance(ys, tuple)
    return (qs == "") if ys == () else strict_prefix(cons2str(ys[0]), qs) and prefix(qs, cons2str(ys))


def format_table(rows, headings=None):
    def fmt(x):
        if hasattr(x, '_repr_html_'):
            return x._repr_html_()
        elif hasattr(x, '_repr_svg_'):
            return x._repr_svg_()
        elif hasattr(x, '_repr_image_svg_xml'):
            return x._repr_image_svg_xml()
        else:
            return f'<pre>{html.escape(str(x))}</pre>'

    return (
        '<table>'
        + (
            '<tr style="font-weight: bold;">'
            + ''.join(f'<td>{x}</td>' for x in headings)
            + '</tr>'
            if headings
            else ''
        )
        + ''.join(
            '<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) + ' </tr>' for row in rows
        )
        + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))
