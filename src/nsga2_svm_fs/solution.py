from random import uniform, randint, sample, choice
from numpy import dot, mean, sign
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, roc_auc_score

class Solution:
    """
    Represents a candidate solution in a multi-objective optimization problem,
    typically involving Support Vector Machines (SVM) and Feature Selection (FS).

    This class contains the necessary attributes and methods for evaluating, mutating,
    and comparing solutions in evolutionary algorithms, such as NSGA-II.

    Attributes
    ----------
    solution_id : int
        Unique identifier for the solution.
    solution_data : any
        Holds any additional data related to the solution (if needed).
    features : list
        List of selected features for the solution.
    vectors : list
        List of vectors related to the solution's representation.
    plane_coords : list
        Coordinates of the solution in the decision space.
    plane_term_b : list
        Bias terms for the decision boundary in the SVM (if applicable).
    objective : list
        A list of objective values: [distance, epsilon, cost, mc_pos, mc_neg].
    dominates_list : list
        List of solution IDs that the current solution dominates.
    dominated_by_list : list
        List of solution IDs that dominate the current solution.
    front : int or None
        The Pareto front index assigned to the solution (used in non-dominated sorting).
    num_dominated_by : int
        Number of solutions that dominate the current solution.
    crowding_distance : float
        The crowding distance of the solution, used in NSGA-II for selection.

    Methods
    -------
    __init__(self, solution_id)
        Initializes a new solution with the given ID.
    """

    def __init__(self, solution_id):
        """
        Initializes a new Solution instance.

        Parameters
        ----------
        solution_id : int
            Unique identifier for the solution.
        """


        self.solution_id = solution_id
        self.solution_data = None
        self.features = []  # List to hold selected features
        self.vectors = []  # List to hold vectors for solution representation
        self.plane_coords = []  # Coordinates in the decision plane
        self.plane_term_b = []  # Bias terms for decision boundaries

        # Objectives: [distance, epsilon, cost, mc_pos, mc_neg]
        self.objective = [0] * 5

        # Front and dominance information for sorting in evolutionary algorithms
        self.dominates_list = []
        self.dominated_by_list = []
        self.front = None  # Pareto front index assigned after non-dominated sorting
        self.num_dominated_by = 0
        self.crowding_distance = 0

        # Matrix confusion and metrics
        self.f1 = None
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.conf_matrix = None
        self.kappa = None
        self.auc = None

    def to_dict(self):
        """
        Converts the solution's attributes to a dictionary format, including all relevant information.

        This dictionary format can be used for easier logging, exporting, or further processing
        of the solution's data.

        Returns
        -------
        dict
            A dictionary containing the solution's ID, objectives, feature data, and other
            relevant attributes for easy access and storage.
        """
        return {
            "SOL": self.solution_id,  # Unique identifier for the solution
            "VECTORS": self.vectors,  # List of vectors associated with the solution
            "PLANO_COOR": self.plane_coords,  # List of coordinates in the decision plane
            "PLANO_COOR_B": self.plane_term_b,  # Bias terms for decision boundaries (if applicable)
            "FEATURES": self.features,  # List of selected features for the solution
            "DIST": self.objective[0],  # Distance value of the solution
            "EPS": self.objective[1],  # Epsilon value of the solution
            "COST": self.objective[2],  # Cost value of the solution
            "MC_POS": self.objective[3],  # Positive margin for multi-class classification
            "MC_NEG": self.objective[4],  # Negative margin for multi-class classification
            "DOMINATES": self.dominates_list,  # List of solutions that this solution dominates
            "DOMINATED_BY": self.dominated_by_list,  # List of solutions that dominate this solution
            "NUM_DOMINATED_BY": self.num_dominated_by,  # Number of solutions that dominate this solution
            "FRONT": self.front,  # Pareto front index assigned to the solution
            "CROWD_DIST": self.crowding_distance,  # Crowding distance value for NSGA-II selection
            # "MC": self.conf_matrix,  # Confusion matrix for the solution
            "ACCURACY": self.accuracy,  # Accuracy score for the solution
            "F1": self.f1,  # F1 score for the solution
            "PRECISION": self.precision,  # Precision score for the solution
            "RECALL": self.recall,  # Recall score for the solution
            "KAPPA": self.kappa,  # Cohen's Kappa score for the solution
            "AUC": self.auc  # Area Under the Curve score for the solution
        }

    def __eq__(self, other):
        """
        Compares whether two solutions are structurally equal based on their key attributes.

        This comparison checks if the following attributes of the two solutions are identical:
        - Vectors (representing decision boundaries or features in the solution)
        - Plane coordinates (which represent the coordinates in a multi-dimensional feature space)
        - Features (the set of selected features used in the solution)

        Parameters
        ----------
        other : Solution
            Another solution object to compare with.

        Returns
        -------
        bool
            True if both solutions have identical vectors, plane coordinates, and features; False otherwise.
        """
        return (
                self.vectors == other.vectors and
                self.plane_coords == other.plane_coords and
                self.features == other.features
        )
        

    def generate_random_solution(self, data, inputs, output, num_features):
        """
        Generates a random solution by selecting feature vectors, features, and coordinates.

        This method creates a random solution by:
        - Selecting vectors based on the unique classes in the dataset.
        - Randomly selecting a subset of features from the input list.
        - Generating unique plane coordinates for the solution.

        The generated solution is represented by the selected vectors, features, and plane coordinates.
        The resulting `solution_data` contains a subset of the dataset corresponding to the selected features.

        Parameters
        ----------
        data : pandas.DataFrame
            Full dataset including features and class labels. Used for generating vectors and selecting features.

        inputs : list
            List of candidate input feature names from which a subset will be randomly selected.

        output : str
            Name of the output column (target class) used to create vectors for each class.

        num_features : int
            Number of features to randomly select from the input list.

        Returns
        -------
        None
            The solution instance is modified in-place with the randomly selected vectors, features, plane coordinates,
            and filtered solution data.
        """
        # Select random vectors for each class (class boundaries)
        classes = sorted(data[output].unique())
        self.vectors = [self.get_class_vector(class_, data, output) for class_ in classes]

        # Randomly select 'num_features' without duplicates
        selected_features = sample(inputs, num_features)

        # Sort the selected features for consistency
        self.features = sorted(selected_features)

        # Generate random coordinates for the plane within the range [-1, 1]
        self.plane_coords = [uniform(-1, 1) for _ in range(num_features)]

        # Filter the dataset based on the selected features and vectors
        self.solution_data = data.loc[self.vectors, self.features]

    def evaluate_solution(self, data, output, costs, inputs):
        """
        Evaluates the solution by computing its objective values.

        Parameters
        ----------
        data : pandas.DataFrame
            Full dataset.
        output : str
            Target column name.
        costs : list
            List of feature costs.
        inputs : list
            List of all possible input features.

        Returns
        -------
        int
            1 if evaluation succeeded, 0 if the solution is invalid.
        """
        if sum(self.plane_coords) == 0:
            return 0

        self.construct_planes(data)
        self.calculate_distance_objective()
        self.calculate_epsilon_objective(data, output)
        self.calculate_cost_objective(costs, inputs)

        return 1

    def construct_planes(self, data):
        """
        Constructs the classification planes from the vectors and coordinates.

        The first plane will be for class -1, the second for class 1, and the third will be the intermediate plane.
        """

        # Select the relevant rows and columns in a single operation
        self.solution_data = data.iloc[self.vectors][self.features]

        # Initialize the bias terms for the planes
        self.plane_term_b = []

        # Calculate the plane bias terms for each selected data point
        for i in range(len(self.solution_data)):
            # Dot product between the data point and the coordinates
            independent_term = dot(self.solution_data.iloc[i].values, self.plane_coords)
            self.plane_term_b.append(-independent_term)

        # Add the intermediate plane (mean of the bias terms)
        self.plane_term_b.append(mean(self.plane_term_b))

    @staticmethod
    def get_class_vector(class_, data, output):
        """
        Randomly selects one sample index corresponding to a class.

        Parameters
        ----------
        class_ : int or str
            Class label.
        data : pandas.DataFrame
            Dataset.
        output : str
            Name of the output column.

        Returns
        -------
        int
            Index of a randomly selected instance from the specified class.
        """
        vector = sample(data.loc[data[output] == class_].index.tolist(), 1)[0]
        return vector

    def calculate_distance_objective(self):
        """
        Calculates the distance objective for the solution, which measures the distance between the planes.
        """
        # Calculating the denominator as the Euclidean norm (magnitude) of the plane coordinates
        denominator = norm(self.plane_coords)

        # Calculate the distance between the planes (b0 and b1)
        distance = abs(self.plane_term_b[0] - self.plane_term_b[1])

        # Avoid division by zero
        self.objective[0] = distance / denominator if denominator != 0 else -1



    def calculate_epsilon_objective(self, data, output):
        """
        Calculates the epsilon objective, which measures the number of misclassified points.

        Parameters
        ----------
        data : pandas.DataFrame
            Full dataset.
        output : str
            Name of the output column.
        """
        # Inicializar objetivos
        self.objective[1] = 0  # Epsilon
        self.objective[3] = 0  # Falsos positivos
        self.objective[4] = 0  # Falsos negativos

        # Norm del vector normal al plano
        norm_ = norm(self.plane_coords)

        # Usamos el plano medio como frontera de decisión
        b0, b1, b_mid = self.plane_term_b[0], self.plane_term_b[1], self.plane_term_b[2]
        #Indicates if the plane for class 1 is above the plane for class -1
        b_condition = b0 > b1

        y_true = []
        y_pred = []
        decision_scores = []
        
        for idx, row in data.iterrows():
            class_ = row[output]
            y_true.append(class_)
            # Dot product between plane_coords and feature values
            distance = sum(
                self.plane_coords[i] * row[feature]
                for i, feature in enumerate(self.features)
            )

            # Función de decisión: f(x) = w·x + b
            f_x = distance + b_mid
            pred = 1 if f_x >= 0 else -1
            y_pred.append(pred)
            decision_scores.append(f_x)

            distance += b1 if class_ == 1 else b0
            distance /= norm_

            
            #b_condition = b0 < b1
            if b_condition:
                if class_ == -1 and distance > 0:
                    self.objective[4] += 1
                    self.objective[1] += distance
                elif class_ == 1 and distance < 0:
                    self.objective[3] += 1
                    self.objective[1] += abs(distance)
            else:
                if class_ == -1 and distance < 0:
                    self.objective[4] += 1
                    self.objective[1] += abs(distance)
                elif class_ == 1 and distance > 0:
                    self.objective[3] += 1
                    self.objective[1] += distance
        
        # Calcular métricas de clasificación
        self.conf_matrix = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        self.recall = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
        self.f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1)
        self.kappa = cohen_kappa_score(y_true, y_pred)

        y_true_auc = [(1 if y == 1 else 0) for y in y_true]
        self.auc = roc_auc_score(y_true_auc, decision_scores)             


    def calculate_cost_objective(self, costs, inputs):
        """
        Computes the total cost associated with the selected features.

        Parameters
        ----------
        costs : list
            Cost of each feature (aligned with `inputs`).
        inputs : list
            Complete list of available features.
        """
        self.objective[2] = sum(costs[inputs.index(f)] for f in self.features)

    def dominates_solution(self, other_solution, method):
        """
        Determines whether this solution dominates another based on the specified method.

        Parameters
        ----------
        other_solution : Solution
            The solution to compare against.
        method : str
            Dominance method: 'Dist-Eps', 'Dist-Eps-Cost', 'MC', or 'MC-Cost'.

        Returns
        -------
        bool
            True if this solution dominates the other one, False otherwise.
        """
        
        if method == 'DIST-EPS':
            
            return (
                    self.objective[0] >= other_solution.objective[0] and
                    self.objective[1] <= other_solution.objective[1]
            )
        elif method == 'DIST-EPS-COST':
            return (
                    self.objective[0] >= other_solution.objective[0] and
                    self.objective[1] <= other_solution.objective[1] and
                    self.objective[2] <= other_solution.objective[2]
            )
        elif method == 'MC':
            return (
                    self.objective[3] <= other_solution.objective[3] and
                    self.objective[4] <= other_solution.objective[4]
            )
        elif method == 'MC-COST':
            return (
                    self.objective[3] <= other_solution.objective[3] and
                    self.objective[4] <= other_solution.objective[4] and
                    self.objective[2] <= other_solution.objective[2]
            )
        else:
            raise ValueError("Invalid method for dominance comparison.")

    def compare_dominance(self, solution2, method):
        """
        Compares dominance between this solution and another.

        Parameters
        ----------
        solution2 : Solution
            The other solution to compare.
        method : str
            Comparison method used.

        Returns
        -------
        int
            0 if equal, 1 if this dominates, 2 if dominated.
        """

        if method == 'DIST-EPS':
            if self.objective[0] == solution2.objective[0] and self.objective[1] == solution2.objective[1]:
                return 0
            elif self.objective[0] >= solution2.objective[0] and self.objective[1] <= solution2.objective[1]:
                return 1
            else:
                return 2
        elif method == 'DIST-EPS-COST':
            if (self.objective[0] == solution2.objective[0] and
                self.objective[1] == solution2.objective[1] and
                self.objective[2] == solution2.objective[2]):
                return 0
            elif (self.objective[0] >= solution2.objective[0] and
                  self.objective[1] <= solution2.objective[1] and
                  self.objective[2] <= solution2.objective[2]):
                return 1
            else:
                return 2
        elif method == 'MC':
            if (self.objective[3] == solution2.objective[3] and
                self.objective[4] == solution2.objective[4]):
                return 0
            elif (self.objective[3] <= solution2.objective[3] and
                  self.objective[4] <= solution2.objective[4]):
                return 1
            else:
                return 2
        elif method == 'MC-COST':
            if (self.objective[3] == solution2.objective[3] and
                self.objective[4] == solution2.objective[4] and
                self.objective[2] == solution2.objective[2]):
                return 0
            elif (self.objective[3] <= solution2.objective[3] and
                  self.objective[4] <= solution2.objective[4] and
                  self.objective[2] <= solution2.objective[2]):
                return 1
            else:
                return 2

    def compare_solutions(self, solution):
        """
        Compares two solutions and selects the better one based on their front and number of dominated solutions.

        Parameters
        ----------
        solution : Solution
            The solution to compare.

        Returns
        -------
        Solution
            The better solution after comparison.
        """
        # First, compare the Pareto front
        if self.front != solution.front:
            return self if self.front < solution.front else solution

        # If the fronts are the same, compare the number of solutions dominated
        num_dominated_self = len(self.dominated_by_list)
        num_dominated_solution = len(solution.dominated_by_list)

        if num_dominated_self != num_dominated_solution:
            return self if num_dominated_self < num_dominated_solution else solution

        # If both are the same in front and dominated solutions, break the tie randomly
        return self if randint(0, 1) == 0 else solution

    def mutate_vectors(self, data, output):
        """
        Mutates class vectors by replacing each with a new one from the same class.

        Parameters
        ----------
        data : pandas.DataFrame
            Full dataset.
        output : str
            Name of the target column.
        """

        for i, class_label in enumerate(sorted(data[output].unique())):
            current_vector = self.vectors[i]
            new_vector = self.get_class_vector(class_label, data, output)
            while new_vector == current_vector:
                new_vector = self.get_class_vector(class_label, data, output)
            self.vectors[i] = new_vector

    def mutate_solution(self, p_mutate_individual, p_mutate_feature,
                        p_mutate_coord, coord_mutation_type,
                        data, inputs, output):
        """
        Applies mutation to class vectors, selected features, and plane coordinates.

        Parameters
        ----------
        p_mutate_individual : float
            Probability of mutating each class vector.
        p_mutate_feature : float
            Probability of replacing each selected feature.
        p_mutate_coord : float
            Probability of mutating each coordinate.
        coord_mutation_type : int
            Type of coordinate mutation (0 = scaling, 1 = full replacement).
        data : pandas.DataFrame
            Dataset to draw new vectors from.
        inputs : list
            List of all candidate features.
        output : str
            Name of the class column.
        """
        # Vector mutation
        if uniform(0, 1) < p_mutate_individual:
            self.mutate_vectors(data, output)

        # Feature mutation:
        # If a feature is mutated, it is replaced with a new one not currently in the solution.
        # A new coordinate on the plane is also assigned to the new feature.
        if len(inputs) > len(self.features):
            
            for i, feat in enumerate(self.features):
                available_features = [f for f in inputs if f not in self.features]
                if uniform(0, 1) < p_mutate_feature and available_features:
                    self.features[i] = choice(available_features)
                    self.plane_coords[i] = uniform(-1, 1)

        # Plane coordinate mutation:
        # Two types of coordinate mutations are supported:
        # - Type 0: mutate by a random percentage (scaling)
        # - Type 1: assign a completely random new coordinate
        for i, coord in enumerate(self.plane_coords):
            if uniform(0, 1) < p_mutate_coord:
                if coord_mutation_type == 0:
                    percentage = uniform(0.05, 0.25)
                    if uniform(0, 1) < 0.5:  # Increase
                        coord *= (1 + percentage)
                        if coord > 1:
                            coord /= 10
                    else:  # Decrease
                        coord *= (1 - percentage)
                        if coord < -1:
                            coord /= 10
                    self.plane_coords[i] = coord
                else:
                    self.plane_coords[i] = uniform(-1, 1)



