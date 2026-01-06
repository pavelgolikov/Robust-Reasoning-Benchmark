import sys
import os

# Add substitutions directory to path so we can import modules
sys.path.append('/home/golikovp/Antigravity/Linguistic_traps/experiments/substitutions')

from wrappers.transformation import apply_wrapper_transformation, get_distractive_wrappers_list

text = "John has 5 apples and 3 bananas."
print("Original:", text)

transformed = apply_wrapper_transformation(text, k=1, seed=42)
print("\nTransformed:\n", transformed)

print("\nWrapper List (first 10):", get_distractive_wrappers_list()[:10])
print("Total wrappers:", len(get_distractive_wrappers_list()))
