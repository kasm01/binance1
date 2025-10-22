from load_env import load_environment_variables

def validate_env():
    env_vars = load_environment_variables()
    missing_vars = [key for key, value in env_vars.items() if value is None]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    print("All environment variables loaded successfully.")
