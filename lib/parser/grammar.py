grammar = """
?start: item+

item: embedding
    | WORD
    | function
    | QUOTED_STRING

function: sum_function
    | neg_function
    | norm_function
    | diff_function

sum_function: "sum(" arg ("|" arg)* ")"
neg_function: "neg(" arg ")"
norm_function: "norm(" arg ")"
diff_function: "diff(" arg ("|" arg)* ")"

arg: item+

embedding: "embedding:" WORD

WORD: /[A-Za-z0-9,_-]+/
QUOTED_STRING: /"([^"\\\]*(\\\.[^"\\\]*)*)"|'([^'\\\]*(\\\.[^'\\\]*)*)'/

%import common.WS
%ignore WS
"""
