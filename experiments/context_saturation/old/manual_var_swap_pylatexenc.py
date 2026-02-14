import re
from pylatexenc.latexwalker import (
    LatexWalker, 
    LatexCharsNode, 
    LatexGroupNode, 
    LatexMacroNode, 
    LatexEnvironmentNode, 
    LatexMathNode
)
import string
import random


class RigorousMathSwapper:
    def __init__(self, latex_source):
        self.source = latex_source
        self.walker = LatexWalker(self.source)
        # Parse the entire document into an AST (Abstract Syntax Tree)
        self.nodes, _, _ = self.walker.get_latex_nodes()

    def get_identifiers(self):
        """
        Walks the tree to find all unique identifiers (variables) 
        inside Math or Asymptote environments.
        """
        variables = set()
        
        def _visit(node, context):
            # Context: 'text', 'math', or 'asy'
            
            if node is None:
                return

            if isinstance(node, LatexCharsNode):
                if context == 'math':
                    # Strict filter: Only single letters are variables
                    # We ignore digits, +, =, etc.
                    for char in node.chars:
                        if char.isalpha():
                            variables.add(char)
                elif context == 'asy':
                    # In Asymptote, we only look inside label("...")
                    # Regex to find content inside double quotes
                    labels = re.findall(r'label\s*\(\s*"(.*?)"', node.chars)
                    for label_content in labels:
                        # Recursively parse the label content as math
                        sub_walker = RigorousMathSwapper(label_content)
                        variables.update(sub_walker.get_identifiers())

            elif isinstance(node, LatexMathNode):
                # Standard $...$ or \[...\]
                for child in node.nodelist:
                    _visit(child, 'math')

            elif isinstance(node, LatexEnvironmentNode):
                new_context = 'text'
                env_name = node.environmentname
                
                # Check for Asymptote
                if env_name == 'asy':
                    new_context = 'asy'
                # Check for math environments (equation, align, etc.)
                elif env_name in ['equation', 'align', 'gather', 'eqnarray']:
                    new_context = 'math'
                
                for child in node.nodelist:
                    _visit(child, new_context)

            elif isinstance(node, LatexGroupNode):
                # {...} inherits current context
                for child in node.nodelist:
                    _visit(child, context)

            elif isinstance(node, LatexMacroNode):
                # Handle arguments of macros (e.g. \frac{x}{y})
                for arg in node.nodeargs:
                    if arg is not None:
                        # Standard args usually inherit context, 
                        # but \text{} switches back to text mode.
                        arg_context = context
                        if node.macroname in ['text', 'mbox']:
                            arg_context = 'text'
                        
                        # Some macros like \sqrt are explicitly math
                        if node.macroname in ['sqrt', 'frac']:
                             # If we were in text, these force math, 
                             # but usually they are already inside math delimiters.
                             pass 

                        # Recurse into the argument's content
                        # nodeargs gives a list of nodes, we need to visit the content
                        _visit(arg, arg_context)

        # Start the traversal
        for node in self.nodes:
            _visit(node, 'text')
            
        return sorted(list(variables))

    def substitute(self, substitution_map):
        """
        Reconstructs the LaTeX string from the AST, swapping variables
        according to the map ONLY inside math/asy contexts.
        """
        
        def _reconstruct(node, context):
            if node is None:
                return ""

            # 1. Handle Text/Chars
            if isinstance(node, LatexCharsNode):
                original_text = node.chars
                
                if context == 'math':
                    # Character-by-character replacement to preserve spacing/punctuation
                    new_text = ""
                    for char in original_text:
                        if char in substitution_map and char.isalpha():
                            new_text += substitution_map[char]
                        else:
                            new_text += char
                    return new_text
                
                elif context == 'asy':
                    # Complex handling for [asy] blocks
                    # We use a callback function with regex to only target label contents
                    def replace_label_content(match):
                        # match.group(1) is the content inside quotes: e.g. "$x$"
                        # We recursively run the substitution on this mini-string
                        inner_swapper = RigorousMathSwapper(match.group(1))
                        # But wait! The inner content might be "$x$". 
                        # The swapper handles delimiters automatically.
                        return 'label("' + inner_swapper.substitute(substitution_map) + '"'
                    
                    # Regex finds label("... and passes it to replace_label_content
                    # We match label followed by optional space, parenthesis, quote
                    return re.sub(r'label\s*\(\s*"(.*?)"', replace_label_content, original_text)

                else:
                    # Text mode: return exactly as is
                    return original_text

            # 2. Handle Math Delimiters ($...$)
            elif isinstance(node, LatexMathNode):
                content = "".join([_reconstruct(n, 'math') for n in node.nodelist])
                return node.latex_verbatim().replace(
                    "".join([n.latex_verbatim() for n in node.nodelist]), 
                    content
                )
                # Note: The above replace is a lazy way to keep the delimiters ($ or \[). 
                # A more rigorous way is manual reconstruction:
                # return node.delimiters[0] + content + node.delimiters[1]

            # 3. Handle Environments (\begin{...} ... \end{...})
            elif isinstance(node, LatexEnvironmentNode):
                new_context = 'text'
                if node.environmentname == 'asy':
                    new_context = 'asy'
                elif node.environmentname in ['equation', 'align', 'gather']:
                    new_context = 'math'
                
                content = "".join([_reconstruct(n, new_context) for n in node.nodelist])
                
                # Rebuild environment structure
                return f"\\begin{{{node.environmentname}}}{content}\\end{{{node.environmentname}}}"

            # 4. Handle Macros (\frac, \sin, etc)
            elif isinstance(node, LatexMacroNode):
                # Reconstruct arguments
                args_str = ""
                for arg in node.nodeargs:
                    if arg is None:
                        continue
                    
                    # Check for context switching macros
                    arg_context = context
                    if node.macroname in ['text', 'mbox']:
                        arg_context = 'text'
                        
                    # Recurse
                    # Note: arg is usually a LatexGroupNode or chars
                    args_str += _reconstruct(arg, arg_context)
                
                # We can't just return \name{args} because some macros use [] or no braces.
                # Use latex_verbatim() for the macro name/structure but replace args?
                # Actually, simplest rigour:
                # Rebuild: \name + arguments
                # But to preserve whitespace after macro? 
                # Pylatexenc stores the macro name. We just append the processed args.
                # However, nodeargs includes the braces { }.
                return f"\\{node.macroname}" + args_str

            # 5. Handle Groups ({...})
            elif isinstance(node, LatexGroupNode):
                content = "".join([_reconstruct(n, context) for n in node.nodelist])
                # We must wrap in braces (or whatever delimiters existed, but usually {})
                return f"{{{content}}}"

            return ""

        # Root level traversal
        result = []
        for node in self.nodes:
            result.append(_reconstruct(node, 'text'))
        
        return "".join(result)


def generate_substitution_map(found_identifiers, distractor_vars=None, seed=42):
    """
    Automatically creates a mapping from the variables found in the problem
    to a target set of variables (e.g., those used in your distractors).
    """
    
    # 1. Define Target Pools (The variables you want to see in the final output)
    # If you have specific variables from your distractor generation, pass them in `distractor_vars`.
    # Otherwise, we use defaults.
    
    if distractor_vars:
        # Assuming distractor_vars is a dict: {'lower': ['a','b'], 'upper': ['X','Y'], ...}
        target_lower = distractor_vars.get('lower', [])
        target_upper = distractor_vars.get('upper', [])
        target_names = distractor_vars.get('names', [])
    else:
        # throw an error
        raise ValueError("distractor_vars must be provided")
        exit(1)
        # # DEFAULT POOLS
        # # We remove 'l', 'o' to avoid confusion with '1', '0' in math problems
        # target_lower = [c for c in string.ascii_lowercase if c not in ['l', 'o', 'e', 'i']]
        # target_upper = [c for c in string.ascii_uppercase if c not in ['I', 'O']]
        # target_names = [
        #     "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", 
        #     "Heidi", "Ivan", "Judy", "Mallory", "Oscar", "Peggy", "Sybil", 
        #     "Trent", "Walter"
        # ]

    # 2. Ensure Determinism (Rigour)
    # We shuffle the pools so we don't always map x->a. 
    # Use a fixed seed if you want the same mapping for the same problem every time.
    rng = random.Random(seed)
    rng.shuffle(target_lower)
    rng.shuffle(target_upper)
    rng.shuffle(target_names)

    sub_map = {}

    # 3. Mapping Logic
    # Helper to map a list of source vars to a list of target vars
    def map_category(source_list, target_pool):
        for i, original in enumerate(source_list):
            # If we run out of target variables, we must stop or recycle.
            # Ideally, your distractor pool is larger than the problem variables.
            if i < len(target_pool):
                target = target_pool[i]
                
                # Collision Check:
                # If the original problem contains "a" and "x", and we map x->a,
                # we have a collision if we don't also map "a" to something else.
                # Since we are re-mapping ALL identified variables, this is safe
                # provided the target pool implies a full replacement.
                sub_map[original] = target
            else:
                # Fallback: keep original if we run out of targets
                print(f"Warning: Ran out of target variables for {original}")
                sub_map[original] = original

    # Execute Mapping
    map_category(found_identifiers['lower'], target_lower)
    map_category(found_identifiers['upper'], target_upper)
    map_category(found_identifiers['names'], target_names)

    return sub_map


# ==========================================
# EXAMPLE USAGE
# ==========================================

# original_problem = r"""
# Let $x$ and $y$ be positive integers such that:
# \begin{equation}
#     x^2 + \frac{12}{y} = \sin(z)
# \end{equation}
# Find the value of $x+y$.

# [asy]
#   unitsize(1cm);
#   draw((0,0)--(5,0));
#   label("$x$", (2.5, 0), S);
#   label("A point $P$", (0,1));
# [/asy]
# """
original_problem = r"""
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she
walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee
shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including
the $t$ minutes spent in the coffee shop.
""" 

# 1. Initialize
processor = RigorousMathSwapper(original_problem)

# 2. Identify Variables (Optional step to help you generate distractors)
vars_found = processor.get_identifiers()

# ==========================================
# INTEGRATION
# ==========================================

# 1. Get identifiers from the previous step
# identifiers = classifier.get_sorted_identifiers()
# Example output from previous step:
# identifiers = {'lower': ['x', 'y'], 'upper': ['P', 'A', 'B'], 'names': ['Alice']}

# 2. (Optional) Define variables used in your fake distractor problems
# If your fake problem used variables m, n, Q, and name "Bob":
distractor_context = {
    'lower': ['x', 'n', 'm', 'k'],  # Your distractors used these
    'upper': ['A', 'B', 'C', 'D'],       # Your distractors used these
    'names': ['Alice', 'Bob', 'Carol', 'David']       # Your distractors used these
}

# 3. Generate the map automatically
auto_map = generate_substitution_map(vars_found, distractor_vars=distractor_context, seed=123)

print("Generated Map:", auto_map)

# 4. Run the substitution
print("\n--- Final Problem ---")
print(classifier.substitute(auto_map))











# print(f"Variables found: {vars_found}") 
# # Output should handle x, y, z, P. It will ignore 'A', 'point', 'Let'.

# # 3. Define your Map (e.g., from Distractors)
# # Let's say your distractor problem used 'a', 'b', 'c', 'Q'
# substitution_map = {
#     'x': 'a',
#     'y': 'b',
#     'z': 'c',
#     'P': 'Q'
# }

# # 4. Execute Substitution
# new_problem = processor.substitute(substitution_map)

# print("\n--- Transformed Problem ---")
# print(new_problem)