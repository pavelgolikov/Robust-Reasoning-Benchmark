import random


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


def generate_system(terms, cur_var_ind):
    """
    Generate a system
    """
    system_description = ""
    # select domain of the system
    
    # determine the number of variables this system description will use
    num_terms = 0
    cur_term_ind = (cur_term_ind + num_terms) % len(terms)

    return system_description, 



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

