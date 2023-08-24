from lark import Lark, Tree, Token, Transformer

# Define the grammar
# grammar = """
# start: token*
#
# token: EMBEDDING
#      | function
#      | arith_action
#      | nudge_action
#      | OPERATOR
#      | TEXT
#
# EMBEDDING: "embedding:" /[^\s]+/ WS?
# function: FUNC_NAME "(" [token ("," token)*] ")" WS?
# arith_action: "<" WS? action_content* ">" WS?
# nudge_action: "[" WS? action_content* "]" WS?
# action_content: EMBEDDING (":" EMBEDDING ":" NUMBER)? WS?
# OPERATOR: ("->" | "+" | "-") WS?
# TEXT: /[a-zA-Z_]\w*/ WS?
# FUNC_NAME: /[a-z_]\w*/
#
# NUMBER: /\d+(\.\d+)?/
#
# %import common.WS
# %ignore WS
# """


# """
# start: token*
#
# token: EMBEDDING
#     //  | function
#      | arith_action
#     //  | nudge_action
#     //  | OPERATOR
#      | TEXT
#
# EMBEDDING: "embedding:" /[^\s]+/ WS?
# // function: FUNC_NAME "(" [token ("," token)*] ")" WS?
# arith_action: "<" arith_content* ">" WS?
# // nudge_action: "[" WS? action_content* "]" WS?
# arith_content: token ":" (arith_ops token)* WS?
# arith_ops: ("+" | "-")
# // OPERATOR: ("->" | "+" | "-") WS?
# TEXT: /[a-zA-Z_]\w*/ WS?
# // FUNC_NAME: /[a-z_]\w*/
#
# NUMBER: /\d+(\.\d+)?/
#
# %import common.WS
# %ignore WS"""

grammar = """
?start: item+

item: embedding
    | WORD
    | function
    | QUOTED_STRING

value: WORD
     | embedding
     | function
     | QUOTED_STRING

function: sum_function
    | neg_function

sum_function: "sum(" value ("|" value)* ")"
neg_function: "neg(" value ")"

embedding: "embedding:" WORD



WORD: /[A-Za-z0-9_-]+/
QUOTED_STRING: /"([^"\\\]*(\\\.[^"\\\]*)*)"|'([^'\\\]*(\\\.[^'\\\]*)*)'/

%import common.WS
%ignore WS
"""

def flatten_tree(tree):
    if isinstance(tree, Token):
        return [str(tree)]
    else:
        return [str(tree.data)] + sum([flatten_tree(child) for child in tree.children], [])

# Initialize the parser
parser = Lark(grammar, start='start', parser='earley')

sample_texts = [
    # "A cat embedding:sdfds sum(rabbit|sum(embedding:sdfds|rose|neg(car))) cat",
    "A cat sum(rabbit|sum(embedding:sdfds|rose|neg(car))) cat",
    # "embedding:scj0_jg [<cat:+embedding:scj0_jg>:<water:+mountain>:0.5] is the embedding:scj0_jg at a <dog:+[embedding:scj0_jg cat:embedding:scj0_jg:0.1]-<water:+ocean>>",
]

class MyTransformer(Transformer):
    # def WORD(self, items):
    #     return items

    def item(self, items):
        print(items)
        return items

    # def value(self, items):
    #     for item in items:
    #         print(item)
    #     return items

    # def embedding(self, items):
    #     return items
    #
    # def sum_function(self, items):
    #     return items
    #
    # def neg_function(self, items):
    #     return items


# Parse the sample text
for sample_action_text in sample_texts:
    parsed_tree = parser.parse(sample_action_text)
    MyTransformer().transform(parsed_tree)
    # print(parsed_tree.pretty())
    print(flatten_tree(parsed_tree))

# Flatten the tree into a list of tokens

