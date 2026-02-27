class ExpSubtypes:
    """Returns StimParserNew subtypes dicts for known beh_type presets.

    Each beh_type has a corresponding static method that builds and returns
    the subtypes dict.  New beh_types are added by:
      1. Writing a ``_<name>`` static method that returns a dict.
      2. Registering the name in ``_DISPATCH`` inside ``get()``.

    Usage
    -----
    subtypes = ExpSubtypes(beh_type='visual_pavlov').get()

    Parameters
    ----------
    beh_type : str
        Experiment type preset.  Currently recognised values:
        'visual_pavlov', 'auditory_pavlov'.
    """

    def __init__(self, beh_type='visual_pavlov'):
        self.beh_type = beh_type

    def get(self):
        """Return the subtypes dict for self.beh_type.

        Returns
        -------
        dict
            Subtypes dict suitable for passing directly to StimParserNew
            as the ``subtypes`` argument.

        Raises
        ------
        ValueError
            If self.beh_type is not a recognised preset.
        """
        _dispatch = {
            'visual_pavlov':   self._visual_pavlov,
            'auditory_pavlov': self._auditory_pavlov,
        }
        if self.beh_type not in _dispatch:
            raise ValueError(
                f"Unknown beh_type: {self.beh_type!r}. "
                f"Known presets: {list(_dispatch)}")
        return _dispatch[self.beh_type]()

    @staticmethod
    def _visual_pavlov():
        """Subtypes for a visual Pavlovian conditioning task.

        Trials are segregated by reward probability (0, 0.5, 1) and, within
        the 50 % condition, by reward outcome and anticipatory licking.
        All filters operate on block-file event variables.
        """
        return {
            '0': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.0},
            },
            '0.5': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.5},
            },
            '1': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 1.0},
            },
            '0.5_rew': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.5,
                            'isRewardGivenValues': 1},
            },
            '0.5_norew': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.5,
                            'isRewardGivenValues': 0},
            },
            '0.5_prelick': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.5,
                            'anticipatoryLickValues': lambda x: x > 0},
            },
            '0.5_noprelick': {
                'parent': None,
                'filters': {'rewardProbabilityValues': 0.5,
                            'anticipatoryLickValues': lambda x: x == 0},
            },
        }

    @staticmethod
    def _auditory_pavlov():
        """Subtypes for an auditory Pavlovian conditioning task.

        No subtypes are defined by default; trials are segregated only by
        the base parsed_param (stimulusType).
        """
        return {}
