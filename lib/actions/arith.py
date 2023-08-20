from custom_nodes.ClipStuff.lib.actions.base import Action


class ArithAction(Action):
    START_CHAR = '<'
    END_CHAR = '>'

    def __init__(self, base_segment: str, ops_str: str):
        self.base_segment = base_segment
        self.ops = self.process_ops_string(ops_str)

    @classmethod
    def process_ops_string(cls, ops_string):
        supported_ops = ['+', '-']
        # dict[[Union[Literal['add'], Literal['subtract']]], str]
        ops_dict = {'+': [], '-': []}
        buff = ''
        curr_op_char = ''
        for char in ops_string:
            if char in supported_ops:
                # We have a buffer
                if buff != '':
                    # Add op string
                    ops_dict[curr_op_char] += [buff]
                    # Reset buffer
                    buff = ''
                    # Set new current op char
                    curr_op_char = char
                    continue
                else:
                    # No buffer, the start of processing
                    curr_op_char = char
            else:
                # Append char to buffer
                buff += char

        # Add last op to dict
        ops_dict[curr_op_char] += [buff]
        return ops_dict
