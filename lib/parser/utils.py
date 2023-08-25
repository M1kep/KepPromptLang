from lark import Token

from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment


def flatten_tree(tree):
    if isinstance(tree, Token):
        return [str(tree)]
    else:
        return [str(tree.data)] + sum([flatten_tree(child) for child in tree.children], [])


def build_prompt_segment(text: str, tokenizer: SD1Tokenizer) -> PromptSegment:
    split_text = text.split(" ")
    tokens = []
    for word in split_text:
        if word.startswith(tokenizer.embedding_identifier) and tokenizer.embedding_directory is not None:
            embedding_name = word[len(tokenizer.embedding_identifier):].strip('\n')

            get_embed_ret = tokenizer._try_get_embedding(embedding_name)
            embedding = get_embed_ret[0]
            leftover = get_embed_ret[1]
            if embedding is None:
                print(f"warning, embedding:{embedding_name} does not exist, ignoring")
            else:
                if len(embedding.shape) == 1:
                    tokens.append(embedding)
                else:
                    tokens.extend(embedding)

            if leftover != "":
                word = leftover
            else:
                continue
        tokens.extend(tokenizer.tokenizer(word)["input_ids"][1:-1])

    return PromptSegment(text, tokens)
