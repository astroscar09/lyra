from lyra import config
import torch
import numpy as np
from lyra.io_utils import load_trained_model, grab_model_file
import pandas as pd
from lyra.evaluation import sample_from_posterior, grab_percentiles_from_posterior
from pathlib import Path
import yaml

class Lyra():
    """
    A class for performing Bayesian inference on Lyman-alpha galaxy properties
    using pre-trained neural density estimators (SBI models).
    
    This class handles model loading, data validation, posterior sampling, and
    summary statistics generation for galaxy property inference.
    
    Attributes:
        posterior: The loaded SBI posterior model for sampling
        model_key (str): Key identifier for the loaded model
        model_schema (dict): Schema containing model metadata and requirements
        data (np.ndarray): Input observational data
        post_samples (np.ndarray): Posterior samples from the model
        summary_df (pd.DataFrame): Summary statistics (percentiles) of posterior
    """

    def __init__(self, model_file):
        """
        Initialize the Lyra inference engine with a pre-trained model.
        
        Args:
            model_file (str): Filename of the model to load (e.g., 'full_SBI_NPE_Muv_beta.pkl')
                            Must be located in the lyra/models/ directory.
        
        Raises:
            RuntimeError: If the model fails to load properly.
            FileNotFoundError: If the model schema file (model_inputs.yaml) is not found.
        
        Note:
            On initialization, the method will print the model configuration including
            required input features and expected dimensionality.
        """
        
        self.load_model_schema()    
        self.model_key = grab_model_file(model_file)
        self.posterior  = None
        self.summary_df = None
        self.data       = None
        self.post_samples    = None

        
        self.grab_model(model_file)
        self.check_model_is_loaded()
        self.init_display_message(self.model_key)

    
    def __repr__(self):
        """
        Return a string representation of the Lyra object.
        
        Returns:
            str: String showing the class name and loaded model key.
        """
        return f"<LyaInference model_key={self.model_key}>"

    

    def grab_model(self, file):
        """
        Load a pre-trained model from the models directory.
        
        Args:
            file (str): Filename of the model pickle file.
        
        Returns:
            None: Sets self.posterior with the loaded model. Falls back to default
                  model if file is not found.
        
        Raises:
            FileNotFoundError: If neither the specified model nor default model is found.
        """
        # Get the models directory path using __file__
        model_dir = Path(__file__).parent / 'models'
        path = model_dir / file
        
        if path.exists():
        
            posterior, _ = load_trained_model(path)
    
        else:
            print(f'No model file found at: {path}')
            print('Defaulting to using the default model')
            
            default_model = model_dir / 'full_SBI_NPE_Muv_beta.pkl'
            print(f'Defaulting to: {default_model}')
            
            posterior, _ = load_trained_model(default_model)

        config.DEVICE = 'cpu'
        self.posterior = posterior
    
    def check_model_is_loaded(self):
        """
        Verify that a model has been successfully loaded.
        
        Returns:
            None
        
        Raises:
            RuntimeError: If self.posterior is None, indicating model load failure.
        """
        if self.posterior is None:
            raise RuntimeError('Posterior Model failed to load. Check file path!')

    def check_data_type(self, data):
        """
        Convert input data to PyTorch tensor and validate type.
        
        Supports multiple input formats: PyTorch tensors, NumPy arrays, and Pandas DataFrames.
        Automatically converts non-tensor types to float32 tensors on the configured device.
        
        Args:
            data (torch.Tensor, np.ndarray, or pd.DataFrame): Input observational data.
        
        Returns:
            torch.Tensor: Data as a PyTorch tensor (dtype=float32) on the configured device.
        
        Prints:
            Status message indicating the input type and any conversions performed.
        """
        if isinstance(data, torch.Tensor):
            print("This is a PyTorch tensor")
        elif isinstance(data, np.ndarray):

            print("This is a NumPy array, converting to PyTorch Tensor")
        
            data = torch.as_tensor(data, dtype=torch.float32).to(config.DEVICE)
                                                                
        elif isinstance(data, pd.DataFrame):
        
            print("This is a DataFrame, converting to PyTorch Tensor")
        
            data = data.values
            data = torch.as_tensor(data, dtype=torch.float32).to(config.DEVICE)
        
        return data


    def predict(self, data, num_samples):
        """
        Generate posterior samples from the model given observed data.
        
        Validates input data shape and type, then samples from the posterior
        distribution of model parameters.
        
        Args:
            data (array-like): Observed data with shape (N, input_dim) where N is
                             the number of observations and input_dim matches the
                             model's expected dimensionality.
            num_samples (int): Number of posterior samples to draw for each observation.
        
        Returns:
            None: Sets self.data and self.post_samples as attributes.
        
        Raises:
            ValueError: If data dimensionality does not match model expectations.
        
        Note:
            This method is typically called through self.sample() which also
            generates summary statistics.
        """
        x_obs = self.check_data_type(data)
        self.validate_shape(x_obs)
        
        self.data = x_obs.numpy()

        samples = sample_from_posterior(self.posterior, x_obs, num_samples)

        self.post_samples = samples.numpy() #this ensure all the following arithmetics operators will work on the numpy array
        

    def generate_summary(self, samples):
        """
        Generate summary statistics from posterior samples.
        
        Calculates percentiles (2.5%, 16%, 50%, 84%, 97.5%) from the posterior
        samples and stores them in a DataFrame for easy access.
        
        Args:
            samples (np.ndarray): Posterior samples with shape (num_observations, num_samples, num_parameters).
        
        Returns:
            None: Sets self.summary_df as a pandas DataFrame with percentile statistics.
        
        Note:
            The resulting DataFrame has columns for each percentile and rows for
            each parameter.
        """
        l2pt5, l16, med, u84, u97pt5 = grab_percentiles_from_posterior(samples)
    
        percentile_data = {'2.5':  l2pt5, 
                           '16':   l16, 
                           '50':   med, 
                           '84':   u84, 
                           '97.5': u97pt5}

        summary_df = pd.DataFrame(percentile_data)
        
        self.summary_df = summary_df
    
    def sample(self, data, num_samples):
        """
        Perform full inference: generate samples and summary statistics.
        
        This is the main method users should call for inference. It combines
        posterior sampling and summary statistics generation in one step.
        
        Args:
            data (array-like): Observed data with shape (N, input_dim).
            num_samples (int): Number of posterior samples to draw per observation.
        
        Returns:
            None: Results stored in self.data, self.post_samples, and self.summary_df.
        
        Example:
            >>> lyra = Lyra('full_SBI_NPE_Muv_beta.pkl')
            >>> data = np.array([[1.0, 2.0, 3.0]])
            >>> lyra.sample(data, num_samples=1000)
            >>> print(lyra.summary_df)  # View percentile statistics
        """
        self.predict(data, num_samples)
        self.generate_summary(self.post_samples)
    
    def _return_quantities(self):
        """
        Return all inference results as a tuple.
        
        Returns:
            tuple: (input_data, posterior_samples, summary_statistics)
                - input_data (np.ndarray): Original input observations
                - posterior_samples (np.ndarray): Full posterior samples
                - summary_statistics (pd.DataFrame): Percentile summary statistics
        
        Note:
            This is an internal method. Users typically access results via
            self.data, self.post_samples, and self.summary_df directly.
        """
        return self.data, self.post_samples, self.summary_df
    
    def load_model_schema(self, schema_file=None):
        """
        Load model metadata schema from YAML file.
        
        The schema contains model configuration information such as expected input
        features, dimensionality, and other model-specific parameters.
        
        Args:
            schema_file (str or Path, optional): Path to the schema YAML file.
                                                 If None, defaults to 'model_inputs.yaml'
                                                 in the models directory.
        
        Returns:
            None: Sets self.model_schema as a dictionary loaded from YAML.
        
        Raises:
            FileNotFoundError: If the schema file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if schema_file is None:
            # Get the models directory path using __file__
            model_dir = Path(__file__).parent / 'models'
            schema_file = model_dir / "model_inputs.yaml"
        
        with open(schema_file) as f:
            self.model_schema = yaml.safe_load(f)
    
   
    def init_display_message(self, model_key):
        """
        Display model configuration and requirements to the user.
        
        Prints a formatted message showing the loaded model key, expected input
        dimensionality, and required input feature order. This helps users verify
        that their data is properly formatted.
        
        Args:
            model_key (str): The key/identifier of the loaded model in the schema.
        
        Returns:
            None: Prints formatted configuration message to stdout.
        
        Raises:
            KeyError: If model_key is not found in self.model_schema.
        """
        schema = self.model_schema[model_key]
        expected_features = schema["input_features"]
        expected_dim = schema["input_dim"]
        
        feature_list = ", ".join(expected_features)

        message = (
                f"\n[Model Loaded Successfully]\n"
                f"  • Model key detected: {model_key}\n"
                f"  • Expected input dimensionality: {expected_dim}\n"
                f"  • Required input feature order:\n"
                f"    [{feature_list}]\n\n"
                f"Please ensure that your input data columns follow this exact order\n"
                f"and have shape (N, {expected_dim}) before running inference.\n"
            )

        print(message)

    def validate_shape(self, data):
        """
        Validate that input data has the correct dimensionality.
        
        Checks that the second dimension (feature dimension) of the input data
        matches the model's expected input dimensionality as specified in the schema.
        
        Args:
            data (torch.Tensor or np.ndarray): Input data with shape (N, input_dim).
        
        Returns:
            None
        
        Raises:
            ValueError: If data.shape[1] does not match the expected dimensionality.
        
        Note:
            This is called automatically by self.predict() before sampling.
        """
        schema = self.model_schema[self.model_key]
        expected_dim = schema["input_dim"]
        
        if data.shape[1] != expected_dim:
            raise ValueError(
                              f"Expected input dim {expected_dim}, got {data.shape[1]}. Plase check input data.#"
                            )

