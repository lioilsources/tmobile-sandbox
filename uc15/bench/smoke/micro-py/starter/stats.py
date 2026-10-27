def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]
