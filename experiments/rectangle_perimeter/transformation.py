import math


def apply_rectangle_perimeter(text):
    """
    Maps text onto the perimeter of a rectangle, written clockwise from the top-left.

    The characters are placed along the edges of a flat rectangle:
      1. Top edge:    left → right
      2. Right edge:  top → bottom
      3. Bottom edge: right → left
      4. Left edge:   bottom → top

    Interior cells are filled with spaces. The result is a multi-line string
    representing the rectangle grid.

    The rectangle is chosen to be wide and flat (width ~4x height) to keep
    the prompt compact.
    """
    if not text:
        return ""

    n = len(text)

    # Choose rectangle dimensions so that the perimeter fits the text.
    # Perimeter of W×H rectangle = 2*W + 2*H - 4  (for H >= 2)
    # We want a wide, flat shape: target W ≈ 4*H
    # Solving: 2*(4H) + 2*H - 4 = n  →  10H = n + 4  →  H = (n+4)/10
    # Minimum H is 2 (need at least top and bottom rows).
    H = max(3, math.ceil((n + 4) / 10))
    # Derive W from perimeter: 2W + 2H - 4 = n  →  W = (n - 2H + 4) / 2
    W = max(2, math.ceil((n - 2 * H + 4) / 2))

    perimeter = 2 * W + 2 * H - 4

    # Pad text with spaces if shorter than perimeter
    padded = text.ljust(perimeter)

    # Build a 2D grid filled with spaces
    grid = [[' ' for _ in range(W)] for _ in range(H)]

    # Walk the perimeter clockwise and place characters
    idx = 0

    # 1. Top edge: row 0, col 0 → W-1
    for c in range(W):
        grid[0][c] = padded[idx]
        idx += 1

    # 2. Right edge: col W-1, row 1 → H-1
    for r in range(1, H):
        grid[r][W - 1] = padded[idx]
        idx += 1

    # 3. Bottom edge: row H-1, col W-2 → 0
    for c in range(W - 2, -1, -1):
        grid[H - 1][c] = padded[idx]
        idx += 1

    # 4. Left edge: col 0, row H-2 → 1
    for r in range(H - 2, 0, -1):
        grid[r][0] = padded[idx]
        idx += 1

    # Render grid as text
    rows = ["".join(row) for row in grid]
    return "GRID START\n" + "\n".join(rows) + "\nGRID END"


def reverse_rectangle_perimeter(text):
    """
    Reads text from the rectangle grid by traversing the perimeter clockwise
    from the top-left corner, recovering the original message.
    """
    if not text:
        return ""

    lines = text.strip().split('\n')
    # Filter out start/end tags
    grid_rows = [
        line for line in lines
        if line.strip() not in ("GRID START", "GRID END", "")
    ]

    if not grid_rows:
        return ""

    H = len(grid_rows)
    W = max(len(row) for row in grid_rows)

    # Pad rows to uniform width (in case of trailing space stripping)
    grid_rows = [row.ljust(W) for row in grid_rows]

    result = []

    # 1. Top edge: row 0, col 0 → W-1
    for c in range(W):
        result.append(grid_rows[0][c])

    # 2. Right edge: col W-1, row 1 → H-1
    for r in range(1, H):
        result.append(grid_rows[r][W - 1])

    # 3. Bottom edge: row H-1, col W-2 → 0
    for c in range(W - 2, -1, -1):
        result.append(grid_rows[H - 1][c])

    # 4. Left edge: col 0, row H-2 → 1
    for r in range(H - 2, 0, -1):
        result.append(grid_rows[r][0])

    return "".join(result).rstrip()
