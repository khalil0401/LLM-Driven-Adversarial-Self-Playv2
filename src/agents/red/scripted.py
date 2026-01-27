class ScriptedRedAgent:
    """
    Executes a pre-defined attack scenario (e.g., SWaT Attack 1).
    Attack: Turn off Pump P1 at t=50 to t=100.
    """
    def __init__(self, attack_id="attack_1"):
        self.attack_id = attack_id
        
    def get_action(self, step_count: int, valid_actions: list) -> dict:
        """
        Returns a dictionary of overrides for the environment.
        """
        overrides = {}
        
        if self.attack_id == "attack_1":
            # "Stop P1 during high demand"
            if 50 <= step_count <= 100:
                overrides['P1'] = 0 # Force P1 Off
        
        elif self.attack_id == "attack_2":
            # "Open Valve 1 causing overflow"
            if 200 <= step_count <= 250:
                overrides['MV1'] = 1
                
        return overrides
