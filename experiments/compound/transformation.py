def apply_compound(target_problem, pre_target_problems):
    """
    Prepends a list of unrelated pre_target_problems before the target_problem.
    Both pre_target_problems and target_problem are numbered.
    """
    lines = ["Solve these completely unrelated math problems. For each problem put your final answer within \\boxed{}.\n"]
    
    for i, problem in enumerate(pre_target_problems):
        lines.append(f"Problem {i+1}:\n{problem}\n")
    
    lines.append(f"Problem {len(pre_target_problems)+1}:\n{target_problem}")
    
    return "\n".join(lines)
