
from experiments.wrappers.transformation import reverse_wrappers

def test_wrapper_collision():
    # Sample 4221 failure case
    # Original: We know that $f^2(1)+f(1)$ divides $4$
    # Reversed reported: We know that $f^1+f(1)$ divides $4$
    
    # Transformed snippet from report:
    # $4(f)^2(2)(6(1))+10(f)(5(1))$ divides $8(4)$
    
    transformed_line = "We know that $4(f)^2(2)(6(1))+10(f)(5(1))$ divides $8(4)$"
    
    # Full defyn block from report (Sample 4221)
    def_block = 'defyn{let "3(s)" mean "s", let "2(2)" mean "2", let "6(1)" mean "1", let "5(1)" mean "1", let "4(f)" mean "f", let "10(f)" mean "f", let "8(4)" mean "4", let "9(1)" mean "1", let "7(2)" mean "2", let "6(4)" mean "4", let "4(equations)" mean "equations", let "5(1)" mean "1", let "7(f)" mean "f", let "3(1)" mean "1", let "3(1)" mean "1", let "8(f)" mean "f", let "3(p)" mean "p", let "7(number)" mean "number", let "7(1)" mean "1", let "6(1)" mean "1", let "2(f)" mean "f", let "9(p)" mean "p", let "10(2)" mean "2", let "3(p)" mean "p", let "8(1)" mean "1", let "3(p)" mean "p", let "10(2)" mean "2", let "6(p)" mean "p", let "7(p)" mean "p", let "5(1)" mean "1", let "3(1)" mean "1", let "2(f)" mean "f", let "5(p)" mean "p", let "6(2)" mean "2", let "3(p)" mean "p", let "5(4)" mean "4", let "3(p)" mean "p", let "8(2)" mean "2", let "6(2)" mean "2", let "9(p)" mean "p", let "7(2)" mean "2", let "4(2)" mean "2", let "7(p)" mean "p", let "7(1)" mean "1", let "5(2)" mean "2", let "6(1)" mean "1", let "3(2)" mean "2", let "4(1)" mean "1", let "10(1)" mean "1", let "5(f)" mean "f", let "4(p)" mean "p", let "9(f)" mean "f", let "8(1)" mean "1", let "6(2)" mean "2", let "10(p)" mean "p", let "5(1)" mean "1", let "7(2)" mean "2", let "2(4)" mean "4", let "5(p)" mean "p", let "2(4)" mean "4", let "7(3)" mean "3", let "8(p)" mean "p", let "6(8)" mean "8", let "3(2)" mean "2", let "5(p)" mean "p", let "7(8p)" mean "8p", let "5(4)" mean "4", let "9(p)" mean "p", let "8(2)" mean "2", let "9(4)" mean "4", let "4(p)" mean "p", let "6(4)" mean "4", let "4(3)" mean "3", let "5(p)" mean "p", let "10(8)" mean "8", let "10(2)" mean "2", let "6(p)" mean "p", let "8(8p)" mean "8p", let "8(4)" mean "4", let "7(2)" mean "2", let "5(4)" mean "4", let "4(p)" mean "p", let "10(2)" mean "2", let "9(2)" mean "2", let "3(p)" mean "p", let "2(2)" mean "2", let "3(4)" mean "4", let "4(p)" mean "p", let "4(4)" mean "4", let "8(3)" mean "3", let "3(p)" mean "p", let "8(8)" mean "8", let "8(2)" mean "2", let "9(p)" mean "p", let "10(8p)" mean "8p", let "6(4)" mean "4", let "10(4)" mean "4", let "2(p)" mean "p", let "3(2)" mean "2", let "10(2)" mean "2", let "6(p)" mean "p", let "7(2)" mean "2", let "3(2)" mean "2", let "6(3)" mean "3", let "8(p)" mean "p", let "4(5)" mean "5", let "9(2)" mean "2", let "2(p)" mean "p", let "6(4)" mean "4", let "10(1)" mean "1", let "4(0)" mean "0", let "10(p)" mean "p", let "3(p)" mean "p", let "6(1)" mean "1", let "10(p)" mean "p", let "5(1)" mean "1", let "4(1)" mean "1", let "7(f)" mean "f", let "4(p)" mean "p", let "10(p)" mean "p", let "10(1)" mean "1", let "2(1)" mean "1", let "7(f)" mean "f", let "9(p)" mean "p", let "2(p)" mean "p", let "3(integer)" mean "integer", let "7(n)" mean "n", let "6(p)" mean "p", let "5(1)" mean "1", let "2(2)" mean "2", let "5(f)" mean "f", let "3(n)" mean "n", let "3(p)" mean "p", let "9(divides)" mean "divides", let "3(1)" mean "1", let "10(2)" mean "2", let "4(2)" mean "2", let "4(p)" mean "p", let "9(n)" mean "n", let "10(1)" mean "1", let "4(2)" mean "2", let "6(1)" mean "1", let "10(2)" mean "2", let "8(p)" mean "p", let "5(f)" mean "f", let "10(n)" mean "n", let "5(p)" mean "p", let "6(2n)" mean "2n", let "8(f)" mean "f", let "7(n)" mean "n", let "9(f)" mean "f", let "10(n)" mean "n", let "9(2)" mean "2", let "3(n)" mean "n", let "5(2)" mean "2", let "5(f)" mean "f", let "3(n)" mean "n", let "7(n)" mean "n", let "2(f)" mean "f", let "10(n)" mean "n", let "5(1)" mean "1", let "5(2)" mean "2", let "2(p)" mean "p", let "3(integer)" mean "integer", let "2(integer)" mean "integer", let "5(p)" mean "p", let "3(integer)" mean "integer", let "2(1)" mean "1", let "7(0)" mean "0", let "3(f)" mean "f", let "10(n)" mean "n", let "5(n)" mean "n", let "6(f)" mean "f", let "9(n)" mean "n", let "5(n)" mean "n", let "10(solution)" mean "solution", let "4(problem)" mean "problem", let "9(Pierre)" mean "Pierre"}.'

    full_text = transformed_line + "\n\n" + def_block
    
    print(f"Input: {full_text}")
    reversed_text = reverse_wrappers(full_text)
    print(f"Output: {reversed_text}")
    
    if "f^2(1)" in reversed_text:
         print("SUCCESS: Found f^2(1)")
    else:
         print("FAILURE: Did not find f^2(1)")
         print(f"Actual segment: {reversed_text.split('divides')[0]}")

if __name__ == "__main__":
    test_wrapper_collision()
