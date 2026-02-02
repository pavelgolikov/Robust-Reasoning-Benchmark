
class ContextLoadManager:
    def __init__(self, target_fill_level: float, max_model_len: int, overshoot_margin: float = 0.05):
        """
        Args:
            target_fill_level (float): 0.0 to 1.0 (e.g. 0.75 for 75%)
            max_model_len (int): Hard limit of the model (e.g. 262000)
            overshoot_margin (float): How much above target is acceptable before backtracking?
                                      (0.05 means we accept up to 80% if target is 75%)
        """
        self.target_fill_level = target_fill_level
        self.max_model_len = max_model_len
        self.overshoot_margin = overshoot_margin
        
        # Derived limits
        self.target_tokens = int(max_model_len * target_fill_level)
        self.hard_limit_margin = int(max_model_len * (target_fill_level + overshoot_margin))
        
        print(f"[ContextLoadManager] Target: {self.target_tokens} tokens ({target_fill_level:.1%})")
        print(f"[ContextLoadManager] Limit w/ Margin: {self.hard_limit_margin} tokens")

    def should_switch_to_solving(self, current_tokens: int) -> bool:
        """
        Decides if we have reached the target fill level.
        If yes, stop feeding distractors.
        """
        return current_tokens >= self.target_tokens

    def check_turn_outcome(self, total_tokens: int) -> str:
        """
        Evaluates the result of a generation turn.
        Returns:
            "ACCEPT": Keep the turn.
            "BACKTRACK_OVERFLOW": Exceeded HARD model limit (Crash prevention).
            "BACKTRACK_OVERSHOOT": Exceeded Target + Margin (Strategy enforcement).
        """
        if total_tokens > self.max_model_len:
            return "BACKTRACK_OVERFLOW"
            
        if total_tokens > self.hard_limit_margin:
            # We filled too much too fast (e.g. runaway generation).
            # The user wants to "remove that output... and rerun".
            return "BACKTRACK_OVERSHOOT"
            
        return "ACCEPT"
