class LabelGenerator():
    grid = None
    params = None

    def __init__(self, grid, params) -> None:
        self.grid = grid
        self.params = params

    def get_labels(self, method='sigma mu'):
        assert method in ['sigma-mu', 'KDE AE', 'quantile']
        
