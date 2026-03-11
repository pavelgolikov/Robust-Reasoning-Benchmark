import random
from generate_systems_static import generate_math_distractor, lcase_dict, ucase_dict, greek_dict

def generate_example():
    # Gather a pool of variables similar to how the main script does it
    all_vars = lcase_dict['high'] + ucase_dict['high'] + greek_dict['high']
    # Pick 9 unique random variables for one system
    variables = random.sample(all_vars, 9)
    
    # Generate one distractor system
    example_system = generate_math_distractor(variables)
    
    print("=" * 60)
    print("EXAMPLE CONTEXT SATURATION SYSTEM (MATH DISTRACTOR)")
    print("=" * 60)
    print(example_system)
    print("=" * 60)

if __name__ == "__main__":
    generate_example()
