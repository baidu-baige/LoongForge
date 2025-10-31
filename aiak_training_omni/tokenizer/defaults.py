"""default tokenizer"""


MODEL_FAMILY_TO_DEFAULT_TOKENIZER = dict(
    [
        ("DEFAULT", 'HFTokenizer'),
        # add custom tokenizer type here
    ]
)


def get_default_tokenizer(family_name: str):
    """get default tokenizer type for model family"""
    tokenizer_type = MODEL_FAMILY_TO_DEFAULT_TOKENIZER.get(family_name, None)
    if tokenizer_type is None:
        tokenizer_type = MODEL_FAMILY_TO_DEFAULT_TOKENIZER.get("DEFAULT", None)
    
    return tokenizer_type
