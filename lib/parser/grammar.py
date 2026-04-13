grammar = r"""
?start: stmt+

?stmt: assign
     | item

assign: "$" NAME "=" arg ";"

item: embedding
    | WORD
    | generic_function
    | QUOTED_STRING
    | weighted
    | ref

generic_function: FUNC_NAME "(" arg ("|" arg)* ")"

weighted: "(" arg ":" SIGNED_NUMBER ")"

ref: "$" NAME

arg: item+

embedding: "embedding:" WORD
// NAME and WORD overlap on bare identifiers; the earley parser's dynamic lexer
// disambiguates by grammar context (the leading "$" forces NAME). This breaks
// under a basic/contextual lexer, so keep parser="earley" in __init__.py.
FUNC_NAME: /[A-Za-z_-]+/
NAME: /[A-Za-z_][A-Za-z0-9_]*/
WORD: /[A-Za-z0-9,_\.-]+/
QUOTED_STRING: /"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)'/
SIGNED_NUMBER: /-?\d+(\.\d+)?/
COMMENT: /#[^\n]*/

%import common.WS
%ignore WS
%ignore COMMENT
"""
