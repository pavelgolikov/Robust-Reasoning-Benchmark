import random

def generate_random_equality_condition(var1, var2):
    """
    Generates a procedurally generated definition for the Equality Operator (==).
    Returns a tuple: (LaTeX_Formula, English_Description)
    
    The logic relies on Equivalence Relations:
    1. Modular Congruence (do they share a remainder?)
    2. Magnitude/Norm (are they the same size?)
    3. Projection (do they map to the same bucket?)
    4. Distance Constraint (are they close enough?)
    """
    
    # Random constants to inject specific numbers into the formula
    k = random.randint(2, 15)
    m = random.randint(3, 50)
    epsilon = random.choice([0.1, 0.5, 1])

    # STRATEGY 1: MODULAR CONGRUENCE
    def strategy_modulo():
        # Formula: a = b (mod k)
        # Variations: Standard, or squared congruence
        if random.random() > 0.5:
            latex = f"({var1} - {var2}) \\equiv 0 \\pmod{{{k}}}"
            desc = f"the difference ({var1} - {var2}) is divisible by {k}"
        else:
            latex = f"{var1}^2 \\equiv {var2}^2 \\pmod{{{k}}}"
            desc = f"their squares are congruent modulo {k}"
        return latex, desc

    # STRATEGY 2: MAGNITUDE / NORM
    def strategy_magnitude():
        # Formula: |a| = |b|
        # Variations: Absolute value, or shifted absolute value
        if random.random() > 0.5:
            latex = f"|{var1}| = |{var2}|"
            desc = f"the absolute value of {var1} equals the absolute value of {var2}"
        else:
            latex = f"|{var1} - {k}| = |{var2} - {k}|"
            desc = f"they are equidistant from the value {k}"
        return latex, desc

    # STRATEGY 3: PROJECTION / QUANTIZATION
    def strategy_quantization():
        # Formula: floor(a / k) = floor(b / k)
        latex = f"\\left\\lfloor \\frac{{{var1}}}{{{k}}} \\right\\rfloor = \\left\\lfloor \\frac{{{var2}}}{{{k}}} \\right\\rfloor"
        desc = f"integer division by {k} yields the same result for both {var1} and {var2}"
        return latex, desc

    # STRATEGY 4: RELAXED EQUALITY (TOLERANCE)
    def strategy_tolerance():
        # Formula: |a - b| <= epsilon
        latex = f"|{var1} - {var2}| \\le {epsilon}"
        desc = f"the distance between {var1} and {var2} is no greater than {epsilon}"
        return latex, desc

    # STRATEGY 5: ALGEBRAIC RELATION
    def strategy_algebraic():
        # Formula: a*b = k (inverse relation) or a+b = k (sum relation)
        # Note: These are rarely reflexive/transitive, which confuses models even more.
        if random.random() > 0.5:
            latex = f"{var1} + {var2} = {m}"
            desc = f"the sum of {var1} and {var2} is exactly {m}"
        else:
            latex = f"({var1} \\cdot {var2}) \\equiv 1 \\pmod{{{k}}}"
            desc = f"{var1} and {var2} are modular multiplicative inverses modulo {k}"
        return latex, desc

    # Pick a random strategy
    options = [
        strategy_modulo, 
        strategy_magnitude, 
        strategy_quantization, 
        strategy_tolerance, 
        strategy_algebraic
    ]
    
    selected_strategy = random.choice(options)
    return selected_strategy()


def generate_random_binary_op(var1, var2):
    """
    Generates a mathematically valid, random algebraic structure.
    Returns a tuple: (LaTeX_Formula, Description)
    """
    # 1. Building Blocks
    random_power = random.randint(-4, 4)
    atoms = [
        f"{var1}",
        f"{var2}",
        f"sqrt({var1})",
        f"sqrt({var2})",
        f"({var1}^{{{random_power}}})",
        f"({var2}^{{{random_power}}})",
        f"({var1}*{var2})",
        f"({var1}+{var2})",
        f"|{var1}-{var2}|"
        f"|{var1}/{var2}|"
    ]
    # print(atoms)
    # exit()
    
    # 2. Connectors
    connectors = ["+", "-", "*", "max", "min", "/", "^"]
    
    # 3. Construct a random depth-2 or depth-3 expression
    # E.g., "(u*v) + (u^2)" or "max(u, v^2) - u"
    
    part_a = random.choice(atoms)
    part_b = random.choice(atoms)
    conn = random.choice(connectors)
    
    if conn in ["max", "min"]:
        formula = f"\\{conn}({part_a}, {part_b})"
    else:
        formula = f"{part_a} {conn} {part_b}"
        
    return formula

def pick_verification_question(domain, terms, sys_index):
    # if domain == "Matrices":
    #     return "Is the system commutative? Formally prove your conclusion."
    # elif domain == "Numbers Module":
    #     return "Is the system associative? Formally prove your conclusion."
    # elif domain == "Polynomials":
    #     return "Is the system distributive? Formally prove your conclusion."
    if len(terms_list) < 3:
        temrs_list[2] = terms_list[0]+"_{0}"
    questions = [
    # Q1: Associativity Verification
    """
    Task: Verify Associativity in System-{sys_index}.
    Consider the specific variables {terms[0]}, {terms[1]}, and {terms[2]}.
    Using the definitions provided, determine if ({terms[0]} * {terms[1]}) * {terms[2]} = {terms[0]} * ({terms[1]} * {terms[2]}).
    
    Steps:
    1. Calculate the LHS: First compute ({terms[0]} * {terms[1]}), then apply * with {terms[2]}.
    2. Calculate the RHS: First compute ({terms[1]} * {terms[2]}), then apply * with {terms[0]}.
    3. Compare the final results using the definition of "=".
    """,

    # Q2: Sequence Expansion
    """
    Task: Calculate Sequence Expansion in System-{sys_index}.
    Using variable {terms[0]} as the starting seed Z_0, define a sequence:
    Z_{{n+1}} = (Z_n + {terms[1]}) * Z_n

    Steps:
    1. Calculate Z_1 explicitly using the formulas for "+" and "*".
    2. Calculate Z_2 using the result of Z_1.
    3. Calculate Z_3 using the result of Z_2.
    4. Check if Z_3 = Z_0.
    """,

    # Q3: Distributivity Check
    """
    Task: Verify Distributivity in System-{sys_index}.
    We must test if the operator "{op_mult}" distributes over "+" for variables {terms[0]}, {terms[1]}, {terms[2]}.
    
    Steps:
    1. LHS: Calculate {terms[0]} * ({terms[1]} + {terms[2]}).
    2. RHS: Calculate ({terms[0]} * {terms[1]}) + ({terms[0]} * {terms[2]}).
    3. Explain any discrepancy between LHS and RHS based on the definitions.
    """,

    # Q4: Commutativity Violation Test
    """
    Task: Measure Commutativity violation in System-{sys_index}.
    Calculate the difference between doing the operation "+" in forward vs reverse order.
    
    Steps:
    1. Calculate Forward: X = {terms[0]} + {terms[1]}.
    2. Calculate Reverse: Y = {terms[1]} + {terms[0]}.
    3. Now, combine them: Result = X * Y.
    """,

    # Q5: Identity Element Search
    """
    Task: Solve for Identity in System-{sys_index}.
    Assume there is an unknown element 'E' such that {terms[0]} + E = {terms[0]}.
    
    Steps:
    1. Write the algebraic equation for "{terms[0]} + E" using the formula defined for "+".
    2. Solve this equation for E in terms of {terms[0]}.
    3. Does this value E work if we replace {terms[0]} with {terms[1]}? Prove it.
    """
]


def generate_system(terms, cur_var_ind, sys_index):
    """
    Generate a system
    """
    system_description = ""
    # select domain of the system
    domains = ["Matrices", "Numbers Module", "Polynomials"]
    domain = random.choice(domains)
    # each system defines 2 terms
    cur_term_ind = (cur_term_ind + 2) % len(terms)
    terms_list = terms[cur_term_ind:cur_term_ind + 2]
    random_int = random.randint(1, 100)
    gen_bin_add = generate_random_binary_op(terms_list[0], terms_list[1])
    gen_bin_mul = generate_random_binary_op(terms_list[0], terms_list[1])
    gen_eq = generate_random_equality_condition(terms_list[0], terms_list[1])
    def_var_1, def_var_2 = terms_list[0], terms_list[1]
    verification_question = pick_verification_question(domain, terms_list, sys_index)
    
    system_dynamic_template = """
        Let us define System-{sys_index}.
        The variables ({def_var_1}, {def_var_2}) are defined as {dynamic_domain_def}.

        The operators are redefined as follows:

        1. DEFINITION OF ADDITION OPERATOR "+":
        For any two elements {def_var_1} and {def_var_2}:
        Formula: {def_var_1} + {def_var_2} = {gen_bin_add}

        2. DEFINITION OF MULTIPLICATION OPERATOR "*":
        For any two elements {def_var_1} and {def_var_2}:
        Formula: {def_var_1} * {def_var_2} = {gen_bin_mul}

        3. DEFINITION OF EQUALITY OPERATOR "=":
        Formula: {def_var_1} = {def_var_2} if and only if {gen_eq}.\n
        """
    
    system_verification_question_template = "For system {sys_index} defined above, {verification_question}"



    return system_description, cur_term_ind



def generate_systems(variables, num_systems):
    """
    Generates a mathematically valid, random mathematical system with given variables to rot the context of the model.
    Returns a list of strings, each string is a definition of a mathematical system and a question to the model about \
    the system.
    """
    systems = []
    cur_sys_ind = 0
    cur_var_ind = 0
    while cur_sys_ind < num_systems:
        # generate a random system
        system_description, cur_var_ind = generate_system(variables, cur_var_ind)
        systems.append(system_description)
        cur_sys_ind += 1
    return systems


if __name__ == "__main__":
    var1 = "u"
    var2 = "v"
    formula = generate_random_binary_op(var1, var2)
    print(formula)

