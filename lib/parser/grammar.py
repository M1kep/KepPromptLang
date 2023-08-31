grammar = """
?start: item+

item: embedding
    | WORD
    | generic_function
    | QUOTED_STRING

generic_function: FUNC_NAME "(" arg ("|" arg)* ")"

arg: item+

embedding: "embedding:" WORD
FUNC_NAME: /[A-Za-z_-]+/
WORD: /[A-Za-z0-9,_\.-]+/
QUOTED_STRING: /"([^"\\\]*(\\\.[^"\\\]*)*)"|'([^'\\\]*(\\\.[^'\\\]*)*)'/

%import common.WS
%ignore WS
"""
