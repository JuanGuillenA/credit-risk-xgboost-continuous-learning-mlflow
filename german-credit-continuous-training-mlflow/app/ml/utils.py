# Defaults for categorical features (reasonable demo defaults)
DEFAULTS = {
    "checking_status": "no checking",
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "savings_status": "no known savings",
    "employment": "1<=X<4",
    "property_magnitude": "real estate",
    "housing": "own",
}

def fill_defaults(payload: dict) -> dict:
    out = payload.copy()
    for k, v in DEFAULTS.items():
        if out.get(k) in (None, "", "null"):
            out[k] = v
    return out
