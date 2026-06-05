from dataclasses import dataclass


@dataclass
class JobError:
    error_type: str
    message: str
    retryable: bool

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }
