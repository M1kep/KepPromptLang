grammar = r"""
?start: item+

item: embedding
    | WORD
    | generic_function
    | QUOTED_STRING
    | weighted

generic_function: FUNC_NAME "(" arg ("|" arg)* ")"

weighted: "(" arg ":" SIGNED_NUMBER ")"

arg: item+

embedding: "embedding:" WORD
FUNC_NAME: /[A-Za-z_-]+/
WORD: /[A-Za-z0-9,_\.-]+/
QUOTED_STRING: /"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)'/
SIGNED_NUMBER: /-?\d+(\.\d+)?/

%import common.WS
%ignore WS
"""
