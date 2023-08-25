grammar = """
?start: item+

item: embedding
    | WORD
    | function
    | QUOTED_STRING

function: sum_function
    | neg_function
    | norm_function

sum_function: "sum(" item* ("|" item)* ")"
neg_function: "neg(" item* ")"
norm_function: "norm(" item* ")"

embedding: "embedding:" WORD

WORD: /[A-Za-z0-9,_-]+/
QUOTED_STRING: /"([^"\\\]*(\\\.[^"\\\]*)*)"|'([^'\\\]*(\\\.[^'\\\]*)*)'/

%import common.WS
%ignore WS
"""
