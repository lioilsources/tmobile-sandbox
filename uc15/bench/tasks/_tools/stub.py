#!/usr/bin/env python3
"""stub.py lang file name1 name2 ... : replace bodies of named functions with a 'not implemented' stub."""
import re, sys
lang, path, names = sys.argv[1], sys.argv[2], sys.argv[3:]
src = open(path).read()
STUB = {'dart': "{\n    throw UnimplementedError('TODO: implement %s');\n  }", 'go': '{\n\tpanic("TODO: implement %s")\n}'}
for name in names:
    m = None
    for cand in re.finditer(r'^[^\n]*?(?<![\w.])' + re.escape(name) + r'\s*\(', src, re.M):
        line = cand.group(0).strip()
        if line.startswith(('return', '//', '///', 'final ', 'if ', 'e.')): continue
        m = cand; break
    if not m: sys.exit(f'not found {name}')
    i = m.end() - 1; depth = 0
    while True:  # skip params
        c = src[i]
        if c == '(': depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0: i += 1; break
        i += 1
    # find body start
    j = i; depth = 0
    while True:
        if src[j] in '([<': depth += 1
        elif src[j] in ')]>' and not src.startswith('=>', j - 1): depth -= 1
        if depth <= 0 and src[j] == '{': kind = 'brace'; break
        if depth <= 0 and src.startswith('=>', j): kind = 'arrow'; break
        j += 1
    k = j; d = 0
    if kind == 'brace':
        while True:
            if src[k] == '{': d += 1
            elif src[k] == '}':
                d -= 1
                if d == 0: k += 1; break
            k += 1
    else:
        while True:
            if src[k] in '({[': d += 1
            elif src[k] in ')}]': d -= 1
            elif src[k] == ';' and d == 0: k += 1; break
            k += 1
    stub = STUB[lang] % name
    if lang == 'dart' and kind == 'arrow': stub = "{\n    throw UnimplementedError('TODO: implement %s');\n  }" % name
    src = src[:j].rstrip() + ' ' + stub + src[k:]
open(path, 'w').write(src)
