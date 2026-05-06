
def apply_rail_fence(text, num_rails):
    """
    Transforms text into a Rail Fence Cipher visual grid.
    Dots (.) are used as place fillers for empty rail spots.
    Spaces in the original text are preserved.
    
    Example (3 rails, text="WE"):
    W . . .
    . E . .
    . . . .
    """
    if not text:
        return ""
    if num_rails <= 1:
        return text

    rails = [['.' for _ in range(len(text))] for _ in range(num_rails)]
    
    row, col = 0, 0
    dir_down = False
    
    for char in text:
        # Place the character
        rails[row][col] = char
        col += 1
        
        # Change direction if at top or bottom rail
        if row == 0 or row == num_rails - 1:
            dir_down = not dir_down
            
        # Move row
        if dir_down:
            row += 1
        else:
            row -= 1
            
    # Construct the grid string
    grid_rows = ["".join(r) for r in rails]
    return "GRID START\n" + "\n".join(grid_rows) + "\nGRID END"

def reverse_rail_fence(text):
    """
    Reverses the Rail Fence Cipher grid back to the original text.
    Assumes standard zigzag pattern (Down, Up...).
    """
    if not text:
        return ""
        
    lines = text.strip().split('\n')
    # Filter out start/end tags
    grid_rows = [line for line in lines if line.strip() != "GRID START" and line.strip() != "GRID END" and line.strip() != ""]
    
    # grid_rows = text.split('\n') # Old logic passed raw text?
    
    num_rails = len(grid_rows)
    if num_rails <= 1:
        return text.replace('\n', '') # Fallback/Error? Or just return text
        
    length = len(grid_rows[0])
    
    result = []
    row, col = 0, 0
    dir_down = False
    
    for i in range(length):
        try:
            char = grid_rows[row][col]
            result.append(char)
            
        except IndexError:
            break # Should not happen if grid is rectangular
            
        col += 1
        if row == 0 or row == num_rails - 1:
            dir_down = not dir_down
            
        if dir_down:
            row += 1
        else:
            row -= 1
            
    return "".join(result)
