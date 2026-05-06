import math

def apply_snake_horizontal(text, height=5):
    """
    Writes the input text into a grid of a specified height using a horizontal 
    "snake" (zigzag) pattern.
    
    The text starts at the top-left (0,0), goes right to (0, W-1),
    then down to (1, W-1), goes left to (1, 0), then down to (2, 0),
    and so on.
    
    Empty positions at the end are filled with equal signs ('=').
    """
    # text = 'Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.'
    if not text:
        return ""

    n = len(text)
    H = height
    W = math.ceil(n / H)
    
    # Initialize grid with None
    grid = [['' for _ in range(W)] for _ in range(H)]
    
    idx = 0
    for r in range(H):
        # Even rows go right, odd rows go left
        if r % 2 == 0:
            cols = range(W)
        else:
            cols = range(W - 1, -1, -1)
            
        for c in cols:
            if idx < n:
                grid[r][c] = text[idx]
                idx += 1
            # else:
                # grid[r][c] = '='
    
    # Render grid
    rendered_rows = ["".join(row) for row in grid]
    # print(rendered_rows)
    # exit()
    return "GRID START\n" + "\n".join(rendered_rows) + "\nGRID END"

def reverse_snake_horizontal(text):
    """
    Recovers the original text from a horizontal snake grid.
    """
    if not text:
        return ""

    lines = text.split('\n')
    # print(lines)
    # exit()
    
    # Identify grid lines between markers
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if "GRID START" in line:
            start_idx = i
        elif "GRID END" in line:
            end_idx = i
            break
            
    if start_idx == -1 or end_idx == -1:
        # Fallback if markers are missing
        grid_rows = [l for l in lines if l.strip() and l not in ("GRID START", "GRID END")]
    else:
        grid_rows = lines[start_idx + 1:end_idx]

    if not grid_rows:
        return ""

    H = len(grid_rows)
    W = max(len(row) for row in grid_rows)
    
    # Pad rows to uniform width
    grid_rows = [row.ljust(W) for row in grid_rows]
    
    result = []
    for r in range(H):
        # Even rows were written left-to-right, odd rows were right-to-left
        if r % 2 == 0:
            cols = range(W)
        else:
            cols = range(W - 1, -1, -1)
            
        for c in cols:
            result.append(grid_rows[r][c])
            
    # Combine and strip trailing '=' padding
    # print(result)
    # exit()
    return "".join(result)
