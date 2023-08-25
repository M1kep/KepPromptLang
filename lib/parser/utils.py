from lark import Token


def flatten_tree(tree):
    if isinstance(tree, Token):
        return [str(tree)]
    else:
        return [str(tree.data)] + sum([flatten_tree(child) for child in tree.children], [])
