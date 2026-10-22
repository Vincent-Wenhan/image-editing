"""Name preprocessing explicitly without changing historical single-reference defaults."""


def reference_preprocessing(kind: str, requested: str = "auto") -> str:
    if requested not in {"auto", "training"}:
        raise ValueError("Reference preprocessing must be auto or training")
    return "training" if kind != "none" or requested == "training" else "diffusers"
