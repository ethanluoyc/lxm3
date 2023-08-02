# TODO(yl): Use a better representation for the job requirements.
class JobRequirements:
    def __init__(self, **kw_resources):
        self.resources = kw_resources or {}
