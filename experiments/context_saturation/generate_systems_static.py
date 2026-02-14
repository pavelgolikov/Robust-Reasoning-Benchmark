import random

# variables_greek = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
# variables_greek_upper = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
# "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"]
# variables_lower = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
# variables_upper = ["X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# variables_AIME_2024 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'a', 
#     'b', 'c', 'f', 'g', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'x', 'y', 'z', 'w']
# vars_total = ["x", "n", "i", "y", "a", "b", "k", "t", "A", "B", "f", "m", "z", "C", "N", "X", "Y", "j", "u", "v", "M",
#     "S", "r", "p", "g", "d", "T", "P", "c", "h", "w", "q", "L", "R", "F", "G", "H", "D", "e", "s", "l", "o", "K", "I",
#     "V", "U", "J", "W", "Z", "Q", "E", "O"]

lcase_high = ['x', 'n', 'i', 't', 'f', 'y', 'a', 'b', 'c', 'e']
lcase_med = ['d', 'g', 'h', 'k', 'm', 'r', 's', 'u', 'v', 'z']
lcase_low = ['j', 'l', 'o', 'p', 'q', 'w']
lcase_dict = {'high': lcase_high, 'med': lcase_med, 'low': lcase_low}

ucase_high = ['A', 'B', 'C', 'P', 'S', 'X', 'Y', 'N', 'M', 'F']
ucase_med = ['D', 'E', 'G', 'H', 'I', 'L', 'R', 'T', 'V', 'Q']
ucase_low = ['J', 'K', 'O', 'U', 'W', 'Z']
ucase_dict = {'high': ucase_high, 'med': ucase_med, 'low': ucase_low}

greek_high = ['\\pi', '\\theta', '\\alpha', '\\beta', '\\Delta', '\\epsilon', '\\lambda', '\\mu', '\\sigma', '\\Sigma']
greek_med = ['\\phi', '\\omega', '\\gamma', '\\rho', '\\tau', '\\delta', '\\eta', '\\psi', '\\nu', '\\xi']
greek_low = ['\\zeta', '\\kappa', '\\chi', '\\iota', '\\upsilon', '\\Omega', '\\Phi', '\\Psi']
greek_dict = {'high': greek_high, 'med': greek_med, 'low': greek_low}




def numeric_equality_conditions(var1, var2):
    # The logic relies on Equivalence Relations:
    # 1. Modular Congruence (do they share a remainder?)
    # 2. Magnitude/Norm (are they the same size?)
    # 3. Projection (do they map to the same bucket?)
    # 4. Distance Constraint (are they close enough?)
    
    # Random constants to inject specific numbers into the formula
    k = random.randint(2, 15)
    m = random.randint(3, 50)
    epsilon = random.choice([0.1, 0.5, 1])

    # STRATEGY 1: MODULAR CONGRUENCE
    def strategy_modulo():
        # Formula: a = b (mod k)
        # Variations: Standard, or squared congruence
        if random.random() > 0.5:
            # latex = f"({var1} - {var2}) \\equiv 0 \\pmod{{{k}}}"
            desc = f"the difference ({var1} - {var2}) is divisible by {k}"
        else:
            # latex = f"{var1}^2 \\equiv {var2}^2 \\pmod{{{k}}}"
            desc = f"their squares are congruent modulo {k}"
        return desc

    # STRATEGY 2: MAGNITUDE / NORM
    def strategy_magnitude():
        # Formula: |a| = |b|
        # Variations: Absolute value, or shifted absolute value
        if random.random() > 0.5:
            # latex = f"|{var1}| = |{var2}|"
            desc = f"the absolute value of {var1} equals the absolute value of {var2}"
        else:
            # latex = f"|{var1} - {k}| = |{var2} - {k}|"
            desc = f"they are equidistant from the value {k}"
        return desc

    # STRATEGY 3: RELAXED EQUALITY (TOLERANCE)
    def strategy_tolerance():
        # Formula: |a - b| <= epsilon
        # latex = f"|{var1} - {var2}| \\le {epsilon}"
        desc = f"the distance between {var1} and {var2} is no greater than {epsilon}"
        return desc

    # STRATEGY 4: ALGEBRAIC RELATION
    def strategy_algebraic():
        # Formula: a*b = k (inverse relation) or a+b = k (sum relation)
        # Note: These are rarely reflexive/transitive, which confuses models even more.
        if random.random() > 0.5:
            # latex = f"{var1} + {var2} = {m}"
            desc = f"the sum of {var1} and {var2} is exactly {m}"
        else:
            # latex = f"({var1} \\cdot {var2}) \\equiv 1 \\pmod{{{k}}}"
            desc = f"{var1} and {var2} are modular multiplicative inverses modulo {k}"
        return desc

    # Pick a random strategy
    options = [
        strategy_modulo, 
        strategy_magnitude, 
        strategy_tolerance, 
        strategy_algebraic
    ]
    
    selected_strategy = random.choice(options)
    return selected_strategy()
    

def matrix_equality_conditions(var1, var2):
    """
    Generates 'made-up' or specific mathematical equality conditions 
    for two square matrices (var1 and var2).
    """
    
    # Constants for specific constraints
    k = random.randint(2, 10)
    epsilon = random.choice([0.01, 0.1, 0.5])
    p = random.choice([1, 2, 'inf', 'fro']) # Matrix norms

    # STRATEGY 1: SPECTRAL PROPERTIES
    def strategy_spectral():
        # Properties related to eigenvalues, trace, and determinant
        choice = random.random()
        if choice < 0.33:
            return f"the trace of {var1} is equal to the trace of {var2}"
        elif choice < 0.66:
            return f"the determinant of {var1} is equal to the determinant of {var2}"
        else:
            return f"{var1} and {var2} share the same characteristic polynomial"

    # STRATEGY 2: STRUCTURAL SIMILARITY
    def strategy_similarity():
        # Equivalence relations in linear algebra
        choice = random.random()
        if choice < 0.5:
            # A = PBP^-1
            return f"{var1} is similar to {var2} (there exists an invertible P such that {var1} = P{var2}P⁻¹)"
        else:
            # A = P^T B P
            return f"{var1} and {var2} are congruent (there exists an invertible P such that {var1} = Pᵀ{var2}P)"

    # STRATEGY 3: SUBSPACE & RANK
    def strategy_subspace():
        # Properties of the range and null space
        choice = random.random()
        if choice < 0.5:
            return f"the column space (range) of {var1} is identical to the column space of {var2}"
        else:
            return f"the rank of {var1} is equal to the rank of {var2}"

    # STRATEGY 4: NORM & DISTANCE
    def strategy_norm():
        # Scalar measurements of the matrix "size"
        norm_names = {'1': 'L1', '2': 'spectral', 'inf': 'infinity', 'fro': 'Frobenius'}
        name = norm_names[str(p)]
        
        if random.random() > 0.5:
            return f"the {name} norm of {var1} is equal to the {name} norm of {var2}"
        else:
            return f"the {name} norm of the difference ({var1} - {var2}) is less than {epsilon}"

    # STRATEGY 5: ALGEBRAIC RELATIONS
    def strategy_algebraic():
        # Commutativity and powers
        choice = random.random()
        if choice < 0.33:
            return f"{var1} and {var2} commute (meaning {var1}{var2} = {var2}{var1})"
        elif choice < 0.66:
            return f"the square of {var1} is equal to the square of {var2}"
        else:
            # Trace modulo
            return f"the trace of ({var1} - {var2}) is a multiple of {k}"

    # Pick a random strategy
    options = [
        strategy_spectral, 
        strategy_similarity, 
        strategy_subspace, 
        strategy_norm,
        strategy_algebraic
    ]
    
    selected_strategy = random.choice(options)
    return selected_strategy()


def polynomial_equality_conditions(var1, var2):
    """
    Generates mathematical equality or equivalence conditions 
    for two polynomials (var1 and var2), typically denoted as P(x) and Q(x).
    """
    
    # Constants for specific constraints
    k = random.randint(-10, 10)
    n = random.randint(1, 5)
    a, b = sorted([random.randint(-5, 5), random.randint(-5, 5)])
    if a == b: b += 1 # Ensure a valid interval
    
    # STRATEGY 1: EVALUATION & ROOTS
    def strategy_evaluation():
        choice = random.random()
        if choice < 0.33:
            return f"{var1}({k}) = {var2}({k}) (they evaluate to the same value at x = {k})"
        elif choice < 0.66:
            return f"{var1}(x) and {var2}(x) share the same set of complex roots"
        else:
            return f"the sum of the roots of {var1}(x) is equal to the sum of the roots of {var2}(x)"

    # STRATEGY 2: CALCULUS (DERIVATIVES & INTEGRALS)
    def strategy_calculus():
        choice = random.random()
        if choice < 0.33:
            return f"the derivative of {var1}(x) is equal to the derivative of {var2}(x) (they differ by a constant)"
        elif choice < 0.66:
            return f"the definite integral of {var1}(x) and {var2}(x) from {a} to {b} are equal"
        else:
            return f"{var1}(x) and {var2}(x) have the same critical points"

    # STRATEGY 3: DIVISIBILITY & MODULAR ARITHMETIC
    def strategy_modular():
        # Using the Polynomial Remainder Theorem logic
        choice = random.random()
        if choice < 0.5:
            return f"{var1}(x) ≡ {var2}(x) (mod x^{n} - 1)"
        else:
            return f"the difference ({var1}(x) - {var2}(x)) is divisible by the linear factor (x - {k})"

    # STRATEGY 4: COEFFICIENTS & DEGREE
    def strategy_coefficients():
        choice = random.random()
        if choice < 0.5:
            return f"the degree of {var1}(x) is equal to the degree of {var2}(x)"
        else:
            # P(1) is the sum of coefficients
            return f"the sum of the coefficients of {var1}(x) is equal to the sum of the coefficients of {var2}(x)"

    # STRATEGY 5: SYMMETRY & TRANSFORMATION
    def strategy_transformation():
        choice = random.random()
        if choice < 0.5:
            return f"{var1}(x) = {var2}(-x) (one is the reflection of the other across the y-axis)"
        else:
            return f"{var1}(x) = {var2}(x + {k}) (one is a horizontal shift of the other)"

    # Pick a random strategy
    options = [
        strategy_evaluation, 
        strategy_calculus, 
        strategy_modular, 
        strategy_coefficients,
        strategy_transformation
    ]
    
    selected_strategy = random.choice(options)
    return selected_strategy()



def generate_random_equality_condition(domain, var1, var2):
    """
    Generates a procedurally generated definition for the Equality Operator (==).
    Returns a tuple: (LaTeX_Formula, English_Description)
    """
    if domain == "Numbers":
        return numeric_equality_conditions(var1, var2)
    elif domain == "Square Matrices":
        return matrix_equality_conditions(var1, var2)
    elif domain == "Polynomials":
        return polynomial_equality_conditions(var1, var2)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def generate_random_binary_op_matrix(var1, var2):
    """
    Generates a random algebraic expression involving two square matrices.
    """

    # 1. Matrix Building Blocks (Atoms)
    # We include unary operations to make the binary operation more complex
    atoms_v1 = [
        f"{var1}",
        f"{var1}^T", # Transpose
        f"{var1}^{{-1}}", # Inverse
        f"tr({var1})I", # Trace scaled identity
        f"exp({var1})", # Matrix exponential
        f"det({var2}){var1}", # Determinant scaled matrix
    ]

    atoms_v2 = [
        f"{var2}",
        f"{var2}^T", # Transpose
        f"{var2}^{{-1}}", # Inverse
        f"tr({var2})I", # Trace scaled identity
        f"exp({var2})", # Matrix exponential
        f"det({var1}){var2}", # Determinant scaled matrix
    ]
    
    # 2. Matrix Connectors
    # Note: Matrix multiplication is denoted by \cdot or juxtaposition
    connectors = ["+", "-", "*", "\\otimes", "\\odot"] 
    # \otimes = Kronecker product, \odot = Hadamard (element-wise) product

    part_a = random.choice(atoms_v1)
    part_b = random.choice(atoms_v2)
    conn = random.choice(connectors)
    formula = f"{part_a} {conn} {part_b}"
        
    return formula


def generate_random_binary_op_polynomial(var1, var2):
    """
    Generates a random algebraic expression involving two polynomials P(x) and Q(x).
    """
    n = random.randint(1, 3)
    k = random.randint(2, 4)
    
    # 1. Polynomial Building Blocks (Atoms)
    atoms_v1 = [
        f"{var1}(x)",
        f"{var1}(x^{n})", # Power
        f"{var1}(-x)", # Reflection
        f"{var1}(x + {k})", # Translation
        f"{var1}(x - {k})", # Translation
        f"{var1}(x * {k})", # Scaling
        f"{var1}(x / {k})", # Scaling
    ]

    atoms_v2 = [
        f"{var2}(x)",
        f"{var2}(x^{n})", # Power
        f"{var2}(-x)", # Reflection
        f"{var2}(x + {k})", # Translation
        f"{var2}(x - {k})", # Translation
        f"{var2}(x * {k})", # Scaling
        f"{var2}(x / {k})", # Scaling
    ]
    
    # 2. Polynomial Connectors
    # Multiplication, addition, and composition
    connectors = ["+", "-", "*", "/", "\\circ"] # \circ is composition: P(Q(x))
    
    # 3. Construct the expression
    part_a = random.choice(atoms_v1)
    part_b = random.choice(atoms_v2)
    conn = random.choice(connectors)
    
    if conn == "\\circ":
        # Standard notation for composition is P(Q(x))
        # We strip the (x) from part_a to make it look like P(Q(x))
        base = part_a.replace("(x)", "")
        formula = f"{base}({part_b})"
    # elif random.random() > 0.8:
    #     # Addition of a modular component (Polynomial Remainder)
    #     formula = f"({part_a} {random.choice(['+', '-'])} {part_b}) \\pmod{{x^{k} + 1}}"
    else:
        formula = f"{part_a} {conn} {part_b}"
        
    return formula


def generate_random_binary_op_numbers(var1, var2):
    """
    Generates a mathematically valid, random algebraic structure.
    Returns a tuple: (LaTeX_Formula, Description)
    """
    # 1. Building Blocks
    random_power = random.randint(-4, 4)
    atoms_v1 = [
        f"{var1}",
        f"sqrt({var1})",
        f"({var1}^{{{random_power}}})",
    ]

    atoms_v2 = [
        f"{var2}",
        f"sqrt({var2})",
        f"({var2}^{{{random_power}}})",
    ]
    
    # 2. Connectors
    connectors = ["+", "-", "*", "max", "min", "/", "^"]
    
    # 3. Construct a random depth-2 or depth-3 expression
    # E.g., "(u*v) + (u^2)" or "max(u, v^2) - u"
    
    part_a = random.choice(atoms_v1)
    part_b = random.choice(atoms_v2)
    conn = random.choice(connectors)
    
    if conn in ["max", "min"]:
        formula = f"{conn}({part_a}, {part_b})"
    else:
        formula = f"{part_a} {conn} {part_b}"
        
    return formula


def generate_random_binary_op(var1, var2, domain):
    if domain == "Numbers":
        return generate_random_binary_op_numbers(var1, var2)
    elif domain == "Square Matrices":
        return generate_random_binary_op_matrix(var1, var2)
    elif domain == "Polynomials":
        return generate_random_binary_op_polynomial(var1, var2)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def pick_verification_question(var1, var2, var3, sys_index):
    questions = [
    # q1: associativity verification
    f"""
verify associativity in system-{sys_index}.
consider the specific variables {var1}, {var2}, and {var3}.
using the definitions provided, determine if ({var1} * {var2}) * {var3} = {var1} * ({var2} * {var3}).
steps:
1. calculate the lhs: first compute ({var1} * {var2}), then apply * with {var3}.
2. calculate the rhs: first compute ({var2} * {var3}), then apply * with {var1}.
3. compare the final results using the definition of "=".
""",

#     # q2: sequence expansion
#     f"""
# calculate sequence expansion in system-{sys_index}.
# using variable {var1} as the starting seed Z_0, define a sequence:
# Z_{{n+1}} = (Z_n + {var2}) * Z_n
# Steps:
# 1. Calculate Z_1 explicitly using the formulas for "+" and "*".
# 2. Calculate Z_2 using the result of Z_1.
# 3. Calculate Z_3 using the result of Z_2.
# 4. Check if Z_3 = Z_0.
# """,

    # Q3: Distributivity Check
    f"""
Verify Distributivity in System-{sys_index}.
We must test if the operator "*" distributes over "+" for variables {var1}, {var2}, {var3}.
Steps:
1. LHS: Calculate {var1} * ({var2} + {var3}).
2. RHS: Calculate ({var1} * {var2}) + ({var1} * {var3}).
3. Explain any discrepancy between LHS and RHS based on the definitions.
""",

    # Q4: Commutativity Violation Test
    f"""
Measure Commutativity violation in System-{sys_index}.
Calculate the difference between doing the operation "+" in forward vs reverse order.
Steps:
1. Calculate Forward: X = {var1} + {var2}.
2. Calculate Reverse: Y = {var2} + {var1}.
3. Now, combine them: Result = X * Y.
""",

    # Q5: Identity Element Search
    f"""
Solve for Identity in System-{sys_index}.
Assume there is an unknown element 'E_i' such that {var1} + E_i = {var1}.
Steps:
1. Write the algebraic equation for "{var1} + E_i" using the formula defined for "+".
2. Solve this equation for E_i in terms of {var1}.
3. Does this value E_i work if we replace {var1} with {var2}? Prove it.
"""]
    return random.choice(questions)


def generate_system(terms, cur_term_ind, sys_index):
    """
    Generate a system
    """
    # select domain of the system
    random_int = random.randint(42, 100)
    domains = ["Square Matrices", f"Numbers", "Polynomials"]
    domain = random.choice(domains)
    # each system defines 2 terms
    cur_term_index = (cur_term_ind + 2) % len(terms)
    terms_list = [terms[(cur_term_index + i) % len(terms)] for i in range(9)]
    gen_bin_add = generate_random_binary_op(terms_list[0], terms_list[1], domain)
    gen_bin_mul = generate_random_binary_op(terms_list[2], terms_list[3], domain)
    gen_eq = generate_random_equality_condition(domain, terms_list[4], terms_list[5])
    # def_var_1, def_var_2, def_var_3, def_var_4, def_var_5, def_var_6, def_var_7, def_var_8, def_var_9 = terms_list
    verification_question = pick_verification_question(terms_list[6], terms_list[7], terms_list[8], sys_index)
    
    system_dynamic_template = f"""\n\n
Let us define System-{sys_index}.
The variables ({terms_list[0]}, {terms_list[1]}, {terms_list[2]}, {terms_list[3]}, {terms_list[4]}, {terms_list[5]}) are defined as {domain}.
The operations on elements of System-{sys_index} are redefined using standard mathematical operations as follows:
1. DEFINITION OF ADDITION OPERATOR on elements of System-{sys_index}: "+":
For any two elements {terms_list[0]} and {terms_list[1]}:
Formula: {terms_list[0]} + {terms_list[1]} = {gen_bin_add}
2. DEFINITION OF MULTIPLICATION OPERATOR on elements of System-{sys_index}: "*":
For any two elements {terms_list[2]} and {terms_list[3]}:
Formula: {terms_list[2]} * {terms_list[3]} = {gen_bin_mul}
3. DEFINITION OF EQUALITY OPERATOR on elements of System-{sys_index}: "=":
Formula: {terms_list[4]} = {terms_list[5]} if and only if {gen_eq}.
    """
    verification_question = f"{verification_question}\n\n"
    prompt = system_dynamic_template + verification_question
    return prompt


def generate_distractor_subsets(lcase_dict, ucase_dict, greek_dict):
    # generates d_h, d_m, and d_l from the provided dictionaries
    # two d_h subsets: one with greek letters and one without
    d_h_no_greek_zip = list(zip(lcase_dict['high'], ucase_dict['high']))
    # flatten the list of tuples
    d_h_no_greek_zip = [item for sublist in d_h_no_greek_zip for item in sublist]
    d_h_greek_zip = list(zip(lcase_dict['high'], ucase_dict['high'], greek_dict['high']))
    # flatten the list of tuples
    d_h_greek_zip = [item for sublist in d_h_greek_zip for item in sublist]
    print(d_h_no_greek_zip)
    print(d_h_greek_zip)
    
    d_m = list(zip(lcase_dict['med'], ucase_dict['med'], greek_dict['med']))
    d_m_no_greek_zip = list(zip(lcase_dict['med'], ucase_dict['med']))
    # flatten the list of tuples
    d_m = [item for sublist in d_m for item in sublist]
    d_m_no_greek_zip = [item for sublist in d_m_no_greek_zip for item in sublist]
    print(d_m_no_greek_zip)
    print(d_m)
    
    d_l = list(zip(lcase_dict['low'], ucase_dict['low'], greek_dict['low']))
    # flatten the list of tuples
    d_l = [item for sublist in d_l for item in sublist]
    print(d_l)


def generate_dset():
    # dset is a distractor set assembled out of 3 basic subsets, which are 
    # d_h - consists of 3 distractors targeting 30 high frequency variables
    # d_m - consists of 3 distractors targeting 30 medium frequency variables
    # d_l - consists of 3 distractors targeting 30 low frequency variables
    # dset = [d_h, d_m, d_h, d_l]
    print("Generating distractor set...")
    


if __name__ == "__main__":
    # system_description = generate_systems_static(vars_total, 2)
    # for i in system_description:
    #     print(i)
    generate_distractor_subsets(lcase_dict, ucase_dict, greek_dict)
