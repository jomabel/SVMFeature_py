class Front:
    """
    Represents a Pareto front in a multi-objective optimization algorithm
    such as NSGA-II.

    Attributes:
        front_id (int): Identifier of the front. For example, front 0 corresponds
                        to the non-dominated set of solutions.
        solutions (list): List of solutions assigned to this front.
                          Each solution is expected to be an object that holds
                          objective values and possibly other attributes relevant
                          to the optimization process.
    """

    def __init__(self, front_id):
        """
        Initializes a new Pareto front with a specific ID and an empty list of solutions.

        Args:
            front_id (int): Unique identifier for the front.
        """
        self.front_id = front_id
        self.solutions = list()


