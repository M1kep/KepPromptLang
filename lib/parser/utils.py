from lark import Token

from comfy.sd1_clip import SDTokenizer

from .prompt_segment import PromptSegment


def flatten_tree(tree):
    if isinstance(tree, Token):
        return [str(tree)]
    return [str(tree.data)] + sum([flatten_tree(child) for child in tree.children], [])


def build_prompt_segment(text: str, tokenizer: SDTokenizer) -> PromptSegment:
    """Tokenize a chunk of plain text into a PromptSegment, expanding `embedding:NAME` refs to tensors."""
    tokens = []
    for word in text.split(" "):
        if word.startswith(tokenizer.embedding_identifier) and tokenizer.embedding_directory is not None:
            embedding_name = word[len(tokenizer.embedding_identifier):].strip("\n")
            embedding, leftover = tokenizer._try_get_embedding(embedding_name)
            if embedding is None:
                print(f"warning, embedding:{embedding_name} does not exist, ignoring")
            elif embedding.shape[1] != tokenizer.embedding_size:
                print(
                    f"warning, embedding:{embedding_name} has size {embedding.shape[1]}, "
                    f"expected {tokenizer.embedding_size}, ignoring"
                )
            else:
                if len(embedding.shape) == 1:
                    tokens.append(embedding)
                else:
                    tokens.extend(embedding)

            if leftover != "":
                word = leftover
            else:
                continue
        # Strip the SOT/EOT bracketing tokens added by the underlying CLIP tokenizer.
        tokens.extend(tokenizer.tokenizer(word)["input_ids"][1:-1])

    return PromptSegment(text, tokens)
