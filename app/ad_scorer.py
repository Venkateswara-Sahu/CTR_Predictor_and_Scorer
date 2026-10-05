"""Historical quality heuristics are retired: Criteo columns are anonymous."""


class AdScorer:
    def __init__(self, *args, **kwargs):
        raise ValueError('Ad quality is not established by anonymous Criteo fields. Use CTR ranking.')
