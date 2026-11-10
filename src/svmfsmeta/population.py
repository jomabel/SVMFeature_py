from random import randint, uniform
from pandas import DataFrame
from .front import Front
from .solution import Solution


class Population:
    """
    Represents a population of individuals (solutions) for a multi-objective evolutionary algorithm.

    Attributes
    ----------
    solutions : list
        List of all individuals (solutions) in the current population.
    fronts : list
        List of lists, where each sublist contains individuals assigned to a particular non-dominated front.
    num_fronts : int or None
        Number of Pareto fronts found during non-dominated sorting.
    solutions_per_front : list or None
        Number of solutions in each front.
    solutions_df : pandas.DataFrame or None
        Tabular representation of all individuals in the population, including objective values and metadata.
    """

    def __init__(self):
        self.solutions = []
        self.fronts = []
        self.num_fronts = None
        self.solutions_per_front = None
        self.solutions_df = None


    def generate_solutions_df(self):
        """
        Generate a pandas DataFrame from the current list of solutions.

        This method converts all individuals in the population into a tabular format
        by calling their `to_dict()` method, then sorts the resulting DataFrame by
        Pareto front (ascending). The result is stored in `self.solutions_df`.

        Returns
        -------
        None
        """
        self.solutions_df = (
            DataFrame.from_records([solution.to_dict() for solution in self.solutions])
            .sort_values(by='FRONT', ascending=True)
            .round(decimals=2)
        )

    def generate_initial_population(
            self,
            population_size: int,
            data,
            inputs: list,
            output: str,
            num_selected_features: int,
            allow_clones: bool,
            method: str
    ):
        """
        Create an initial population of valid, evaluated, and (optionally) unique solutions.

        This method repeatedly generates random candidate solutions, evaluates them,
        and adds them to the population if they are valid. If `allow_clones` is False,
        it ensures that no duplicate solutions are added.

        Parameters
        ----------
        population_size : int
            Target number of individuals in the population.
        data : pandas.DataFrame
            Dataset containing input features and output.
        inputs : list
            Names of input (feature) columns.
        output : str
            Name of the target/output column.
        num_selected_features : int
            Number of features to select in each solution.
        allow_clones : bool
            Whether to allow duplicated (clone) individuals in the population.
        method : str
            The method to use for dominance comparison.

        Returns
        -------
        None
        """
        solution_id = 0

        while len(self.solutions) < population_size:
            solution = Solution(solution_id)
            solution.generate_random_solution(data, inputs, output, num_selected_features)

            if solution.evaluate_solution(data, output, inputs) != 1:
                continue

            if allow_clones or not self.check_clones(solution, method):
                self.solutions.append(solution)
                solution_id += 1

        self.generate_solutions_df()

    def check_clones(self, evaluated_solution: 'Solution', method):
        """
        Check if the given solution is a duplicate (clone) of any existing solution in the population.

        This method compares the provided `evaluated_solution` with each solution in the population.
        It returns `True` if a duplicate is found, otherwise `False`.

        Parameters
        ----------
        evaluated_solution : Solution
            The solution to check for duplication against the current population.
        method : str
            The method to use for dominance comparison.

        Returns
        -------
        bool
            True if the solution is a duplicate (clone) of an existing one, False otherwise.
        """
        for sol in self.solutions:
            if method == 'DIST-EPS':
                sol_obj_a = sol.objective[0]
                sol_obj_b = sol.objective[1]
                eva_obj_a = evaluated_solution.objective[0]
                eva_obj_b = evaluated_solution.objective[1]

            elif method == 'MC':
                sol_obj_a = sol.objective[2]
                sol_obj_b = sol.objective[3]
                eva_obj_a = evaluated_solution.objective[2]
                eva_obj_b = evaluated_solution.objective[3]

            else:
                continue

            if (sol.features == evaluated_solution.features and
                abs(sol_obj_a - eva_obj_a) < 0.1 and
                abs(sol_obj_b - eva_obj_b) < 0.1 and
                len(sol.plane_coords) == len(evaluated_solution.plane_coords) and
                all(abs(a - b) < 0.001 for a, b in zip(sol.plane_coords, evaluated_solution.plane_coords)) and
                sol.vectors == evaluated_solution.vectors):
                return True  # clon encontrado

        return False  # no hay clon

    def fnds(self, method):
        """
        Perform Fast Non-dominated Sorting (FNDS) on the current population.

        This method assigns each individual solution in the population to a Pareto front
        based on dominance relations, which is a key component of NSGA-II. It computes
        the non-dominated sorting and stores the results in `self.fronts`, where each front
        is a list of solutions that are not dominated by any other solution in the population.

        Parameters
        ----------
        method : str
            The method to use for dominance comparison (e.g., 'DIST-EPS', 'MC', etc.).

        Returns
        -------
        None
        """

        self.fronts = []  # List to store the fronts
        population_size = len(self.solutions)
        front = Front(0)

        # Reset dominance-related attributes
        for sol in self.solutions:
            sol.num_dominated_by = 0  # How many solutions dominate this one
            sol.dominates_list = []  # List of solution indices this solution dominates
            sol.dominated_by_list = []  # List of indices that dominate this solution

        # Compare each pair of solutions to establish dominance
        for i in range(population_size):
            for j in range(i + 1, population_size):
                sol_i = self.solutions[i]
                sol_j = self.solutions[j]

                if sol_i.dominates_solution(sol_j, method):
                    sol_j.num_dominated_by += 1
                    sol_i.dominates_list.append(sol_j.solution_id)
                    sol_j.dominated_by_list.append(sol_i.solution_id)
                elif sol_j.dominates_solution(sol_i, method):
                    sol_i.num_dominated_by += 1
                    sol_j.dominates_list.append(sol_i.solution_id)
                    sol_i.dominated_by_list.append(sol_j.solution_id)

            if self.solutions[i].num_dominated_by == 0:
                self.solutions[i].front = 0
                front.solutions.append(self.solutions[i].solution_id)

        self.fronts.append(front)

        # Build subsequent fronts
        front_index = 0
        while self.fronts[front_index].solutions:
            next_front = Front(front_index + 1)
            for sol_index in self.fronts[front_index].solutions:
                sol = self.solutions[sol_index]
                for dominated_index in sol.dominates_list:
                    dominated_sol = self.solutions[dominated_index]
                    dominated_sol.num_dominated_by -= 1
                    if dominated_sol.num_dominated_by == 0:
                        dominated_sol.front = front_index + 1
                        next_front.solutions.append(dominated_index)
            front_index += 1
            self.fronts.append(next_front)

        self.generate_solutions_df()

    def create_new_population(
            self,
            population_size: int,
            p_mutation: float,
            data,
            inputs: list,
            output: str,
            p_mutate_individual: float,
            p_mutate_feature: float,
            p_mutate_coord: float,
            coord_mutation_type: int,
            allow_clones: bool,
            method: str
    ):
        """
        Generate a new population through crossover and mutation.

        This method uses tournament selection to choose two distinct parents, performs
        crossover to generate offspring, and applies mutation based on predefined probabilities.
        The new population is expanded to 2 * population_size. The method ensures that only valid
        and non-clone solutions are added to the population.

        Parameters
        ----------
        population_size : int
            Size of the current population, which will be expanded to 2 * population_size.
        p_mutation : float
            Probability of mutation being applied to an offspring.
        data : pandas.DataFrame
            Dataset containing input features and output.
        inputs : list
            List of input (feature) column names.
        output : str
            Target/output column name.
        p_mutate_individual : float
            Probability of individual mutation occurring.
        p_mutate_feature : float
            Probability of feature mutation occurring.
        p_mutate_coord : float
            Probability of coordinate mutation occurring.
        coord_mutation_type : int
            Type of coordinate mutation to apply (e.g., mutation strategy).
        allow_clones : bool
            Whether to allow duplicate (clone) individuals in the population.
        method : str
            Method to use for dominance comparison during crossover (e.g., 'DIST-EPS', 'MC').

        Returns
        -------
        None
        """
        i = population_size
        max_attempts = 10
        
        while i < 2 * population_size:
            # Select two distinct parents using tournament selection
            father = self.tournament_select_parent(population_size)
            mother = self.tournament_select_parent(population_size)

            # Ensure both parents are distinct
            while father.solution_id == mother.solution_id:
                mother = self.tournament_select_parent(population_size)

            # Perform crossover: create 4 offspring and keep the 2 best
            offspring = self.crossover_solutions(father, mother, i, method, data, inputs, output)

            # Mutate the offspring
            for child in offspring:
                # Check if mutation should be applied
                if uniform(0, 1) < p_mutation:
                    child.mutate_solution(
                        p_mutate_individual,
                        p_mutate_feature,
                        p_mutate_coord,
                        coord_mutation_type,
                        data,
                        inputs,
                        output
                    )
                # Ensure the solution is valid
                while child.evaluate_solution(data, output, inputs) == 0:
                    child.generate_random_solution()

                attempts = 0
                # Check for clones and mutate if needed
                while self.check_clones(child, method) != 0 and allow_clones == 0 and attempts < max_attempts:
                    child.mutate_solution(
                        p_mutate_individual,
                        p_mutate_feature,
                        p_mutate_coord,
                        coord_mutation_type,
                        data,
                        inputs,
                        output
                    )

                    # Ensure the solution is valid
                    validity_attempts = 0
                    while (child.evaluate_solution(data, output, inputs) == 0 and
                           validity_attempts < max_attempts):
                        child.mutate_solution(
                            p_mutate_individual,
                            p_mutate_feature,
                            p_mutate_coord,
                            coord_mutation_type,
                            data,
                            inputs,
                            output
                        )
                        validity_attempts += 1
                    attempts += 1
                
                
                # Lastly, check if the child is valid and not a clone
                if allow_clones == 0 and self.check_clones(child, method) == 0 and child.evaluate_solution(data, output, inputs) != 0:
                    # Add valid child to the population
                    self.solutions.append(child)
                    i = len(self.solutions)


                
       



        self.generate_solutions_df()

    def reduce_population(self, population_size: int, method: str):
        """
        Reduces the population size from 2N to N using non-dominated sorting and crowding distance.

        This method follows these steps:
        1. Adds entire Pareto fronts to the reduced population as long as they fit within the target population size.
        2. If there is remaining space, selects the best individuals from the next front based on their crowding distance.

        Parameters
        ----------
        population_size : int
            The desired population size after reduction.
        method : str
            The method to use for calculating the crowding distance. Options are:
            "DIST-EPS", "MC".

        Returns
        -------
        reduced_population : Population
            A new population object with the reduced size, equal to `population_size`.
        """
        current_front = 0
        reduced_population = Population()
        reduced_population.solutions = []

        # Add entire fronts until the population reaches the target size
        while (len(reduced_population.solutions) + len(self.fronts[current_front].solutions)
               <= population_size):
            front_solutions = [sol for sol in self.solutions if sol.front == current_front]
            reduced_population.solutions.extend(front_solutions)
            current_front += 1

        # If there's remaining space, fill it using crowding distance to select the best individuals
        remaining_needed = population_size - len(reduced_population.solutions)
        if remaining_needed > 0:
            sorted_by_crowding = self.calculate_crowding_distance(method, current_front)
            reduced_population.solutions.extend(sorted_by_crowding[:remaining_needed])

        # Reassign solution indices
        for idx, sol in enumerate(reduced_population.solutions):
            sol.solution_id = idx

        reduced_population.generate_solutions_df()

        return reduced_population

    def select_two_distinct_parents(self, population_size: int):
        """
        Selects two distinct parents using tournament selection.

        This method selects two parents from the population using tournament selection.
        It ensures that the two parents are distinct by comparing their solution IDs,
        retrying if both parents have the same ID.

        Parameters
        ----------
        population_size : int
            The total number of solutions in the population, used to constrain parent selection.

        Returns
        -------
        Tuple[Solution, Solution]
            A pair of distinct parent solutions selected using tournament selection.
        """
        father = self.tournament_select_parent(population_size)
        mother = self.tournament_select_parent(population_size)

        # Ensure that the father and mother are distinct
        while father.solution_id == mother.solution_id:
            mother = self.tournament_select_parent(population_size)

        return father, mother

    def tournament_select_parent(self, population_size: int):
        """
        Selects a parent from the population using tournament selection.

        This method randomly selects two solutions from the population and compares them based on dominance.
        The solution that dominates (or is better) is returned as the selected parent.

        Parameters
        ----------
        population_size : int
            The size of the population from which the parent is selected.

        Returns
        -------
        Solution
            The selected parent solution, which is the dominant one of the two chosen.
        """
        # Randomly select two distinct solutions from the population
        parent1 = self.get_solution_by_id(randint(0, population_size - 1))
        parent2 = self.get_solution_by_id(randint(0, population_size - 1))

        # Ensure that the two solutions are distinct
        while parent1.solution_id == parent2.solution_id:
            parent2 = self.get_solution_by_id(randint(0, population_size - 1))

        # Compare the two solutions and return the dominant one
        return parent1.compare_solutions(parent2)

    def get_solution_by_id(self, solution_id: int):
        """
        Retrieves a solution from the population by its ID.

        This method iterates through the population and returns the solution that
        matches the given solution ID. If no solution with the specified ID is found,
        the method returns None.

        Parameters
        ----------
        solution_id : int
            The ID of the solution to retrieve.

        Returns
        -------
        Solution
            The solution with the matching ID, or None if no solution with the given ID is found.
        """
        # Iterate through the solutions and return the one with the matching ID
        for solution in self.solutions:
            if solution.solution_id == solution_id:
                return solution

        # Return None if no solution with the given ID is found
        return None

    def crossover_solutions(self, father: Solution, mother: Solution, solution_id: int, method: str,
                            data, inputs: list, output: str):
        """
        Performs crossover between two parent solutions (father and mother) to create offspring.

        This method generates four potential offspring by combining features, coordinates, and vectors from
        the parents. It then compares them based on dominance and selects the two best offspring.

        Parameters
        ----------
        father : Solution
            The father solution, which provides some features for crossover.
        mother : Solution
            The mother solution, which provides other features for crossover.
        solution_id : int
            The identifier for the new solutions.
        method : str
            The method used for comparing the offspring to determine the best ones.
        data :
            The dataset used for evaluating solutions.
        inputs : list
            The input data to evaluate the solutions.
        output : str
            The output label for evaluation.

        Returns
        -------
        List[Solution]
            A list containing the two best offspring solutions after crossover and comparison.
        """

        # Give each child its own gene lists so mutation leaves parents and siblings intact.
        # Create four offspring solutions using crossover from the parents
        # First child: father's features and coordinates, father's first vector and mother's second vector
        child1 = Solution(solution_id)
        child1.features = list(father.features)
        child1.plane_coords = list(father.plane_coords)
        child1.vectors = [father.vectors[0], mother.vectors[1]]

        # Second child: mother's features and coordinates, father's first vector and mother's second vector
        child2 = Solution(solution_id + 1)
        child2.features = list(mother.features)
        child2.plane_coords = list(mother.plane_coords)
        child2.vectors = [father.vectors[0], mother.vectors[1]]

        # Third child: father's features and coordinates, mother's first vector and father's second vector
        child3 = Solution(solution_id + 2)
        child3.features = list(father.features)
        child3.plane_coords = list(father.plane_coords)
        child3.vectors = [mother.vectors[0], father.vectors[1]]

        # Fourth child: mother's features and coordinates, mother's first vector and father's second vector
        child4 = Solution(solution_id + 3)
        child4.features = list(mother.features)
        child4.plane_coords = list(mother.plane_coords)
        child4.vectors = [mother.vectors[0], father.vectors[1]]

        # Evaluate the offspring solutions 
        child1.evaluate_solution(data, output, inputs)
        child2.evaluate_solution(data, output, inputs)
        child3.evaluate_solution(data, output, inputs)
        child4.evaluate_solution(data, output, inputs)


        # First round of comparison
        winner1, loser1 = self.select_best_offspring(child1, child2, method)
        winner2, loser2 = self.select_best_offspring(child3, child4, method)

        # Final between both winners
        best, second_candidate = self.select_best_offspring(winner1, winner2, method)

        # Choose second best among remaining three (two losers + second best from final)
        second_pool = [loser1, loser2, second_candidate]
        second_best = second_pool[0]
        for candidate in second_pool[1:]:
            second_best, _ = self.select_best_offspring(second_best, candidate, method)

        # Assign solution IDs
        best.solution_id = solution_id
        second_best.solution_id = solution_id + 1

        return [best, second_best]

    @staticmethod
    def select_best_offspring(child1: Solution, child2: Solution, method: str):
        """
        Compares two offspring and selects the best one based on dominance relations.

        This method uses the concept of dominance in multi-objective optimization to compare the two offspring.
        If one offspring dominates the other, the dominant one is selected. If neither dominates the other,
        the tie is broken randomly.

        Parameters
        ----------
        child1 : Solution
            The first offspring to compare.
        child2 : Solution
            The second offspring to compare.
        method : str
            The multi-objective optimization method ('DIST-EPS', 'MC').

        Returns
        -------
        Tuple[Solution, Solution]
            A tuple with two solutions, where the first one is the better offspring (the winner), and the second one
            is the inferior offspring (the loser).
        """

        # Compare the two offspring based on dominance
        dominance_result = child1.compare_dominance(child2, method)

        # Return the winner and loser based on the dominance result
        if dominance_result == 1:
            return child1, child2  # child1 dominates child2
        elif dominance_result == 2:
            return child2, child1  # child2 dominates child1
        else:
            # If neither dominates, break the tie randomly
            return (child1, child2) if uniform(0, 1) < 0.5 else (child2, child1)

    def calculate_crowding_distance(self, method: str, front_id: int):
        """
        Calculates the crowding distance for each solution in the specified Pareto front.

        The crowding distance is a measure of how close solutions are to each other in the objective space.
        It is used in multi-objective optimization to maintain diversity in the population.

        Parameters
        ----------
        method : str
            The method to use for determining which objectives to consider. Options are:
            - "DIST-EPS" for distance and epsilon
            - "MC" for MC+ and MC-
        front_id : int
            The index of the Pareto front for which to compute crowding distance.

        Returns
        -------
        List[Solution]
            A list of solutions from the specified Pareto front, sorted by crowding distance in descending order.
        """

        # Filter solutions in the given Pareto front
        solutions_front = [s for s in self.solutions if s.front == front_id]
        num_sol = len(solutions_front)

        # Return an empty list if there are no solutions in this front
        if num_sol == 0:
            return []

        # Initialize crowding distance to 0 for all solutions
        for sol in solutions_front:
            sol.crowding_distance = 0

        # Determine the number of objectives based on the selected method
        if method == "DIST-EPS" or method == "MC" :
            num_obj = 2
        else:
            raise ValueError(f"Method '{method}' is not recognized.")

        # Helper function to extract the relevant objectives based on the method
        def get_objectives(solution, method_):
            if method_ == "DIST-EPS":
                return [solution.objective[0], solution.objective[1]]  # [DIST, EPS]
            elif method_ == "MC":
                return [solution.objective[2], solution.objective[3]]  # [MC+, MC-]
            else:
                raise ValueError(f"Method '{method_}' is not recognized.")

        # Calculate crowding distance for each objective
        for obj_idx in range(num_obj):
            # Sort solutions by the current objective value
            solutions_front.sort(key=lambda x: get_objectives(x, method)[obj_idx])

            # Assign extreme crowding distance values to the boundary solutions
            solutions_front[0].crowding_distance = float('inf')
            solutions_front[-1].crowding_distance = float('inf')

            # Calculate the crowding distance for each solution (except the boundaries)
            min_val = get_objectives(solutions_front[0], method)[obj_idx]
            max_val = get_objectives(solutions_front[-1], method)[obj_idx]

            if max_val == min_val:
                continue  # Avoid division by zero if all solutions have the same value for this objective

            for j in range(1, num_sol - 1):
                prev_val = get_objectives(solutions_front[j - 1], method)[obj_idx]
                next_val = get_objectives(solutions_front[j + 1], method)[obj_idx]
                solutions_front[j].crowding_distance += (next_val - prev_val) / (max_val - min_val)

        # Sort solutions by crowding distance in descending order (higher distance is better)
        solutions_front.sort(key=lambda x: x.crowding_distance, reverse=True)

        return solutions_front
