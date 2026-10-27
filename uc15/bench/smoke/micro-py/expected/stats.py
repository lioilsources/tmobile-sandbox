def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
