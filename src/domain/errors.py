class DomainError(Exception):
    pass

class InsufficientDataError(DomainError):
    def __init__(self, message: str = "bad data for model estimation"):
        self.message = message
        super().__init__(self.message)

class InvalidDataError(DomainError):
    def __init__(self, message: str = "invalid data provided"):
        self.message = message
        super().__init__(self.message)

class ModelEstimationError(DomainError):
    def __init__(self,message: str = "model estimtion failed"):
        self.message = message
        
        super().__init__(self.message)