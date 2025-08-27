from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
import json
from pathlib import Path

# Configuration
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_config_dir():
    """Get platform-appropriate config directory."""
    if os.name == 'nt':  # Windows
        config_dir = Path(os.environ.get('APPDATA', Path.home())) / 'microbiome-app'
    else:  # macOS/Linux
        config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'microbiome-app'
    
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_client_secrets_path():
    """Get client secrets file path from environment or default."""
    env_path = os.environ.get('GOOGLE_CLIENT_SECRETS_PATH')
    if env_path:
        return Path(env_path)
    
    # Default to config directory
    return get_config_dir() / 'client_secrets.json'

def get_token_path():
    """Get token file path from environment or default."""
    env_path = os.environ.get('GOOGLE_TOKEN_PATH')
    if env_path:
        return Path(env_path)
    
    # Default to config directory
    return get_config_dir() / 'token.json'

def create_client_secrets_from_env():
    """Create client secrets from environment variables."""
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
    project_id = os.environ.get('GOOGLE_PROJECT_ID', 'your-project-id')
    
    if not client_id or not client_secret:
        raise ValueError(
            "Missing required environment variables. Please set:\n"
            "  GOOGLE_CLIENT_ID\n"
            "  GOOGLE_CLIENT_SECRET\n"
            "  GOOGLE_PROJECT_ID (optional)\n"
            "Or provide client_secrets.json at the path specified by GOOGLE_CLIENT_SECRETS_PATH"
        )
    
    return {
        "installed": {
            "client_id": client_id,
            "project_id": project_id,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost"]
        }
    }

def main():
    """Generate and save OAuth token securely."""
    token_path = get_token_path()
    client_secrets_path = get_client_secrets_path()
    
    print(f"Token will be saved to: {token_path}")
    print(f"Looking for client secrets at: {client_secrets_path}")
    
    creds = None
    
    # Load existing credentials if available
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            print("Loaded existing credentials")
        except Exception as e:
            print(f"Error loading existing credentials: {e}")
            creds = None
    
    # Check if credentials are valid or can be refreshed
    if creds and creds.valid:
        print("Credentials are already valid")
        return
    
    if creds and creds.expired and creds.refresh_token:
        try:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
            
            # Save refreshed credentials
            with open(token_path, 'w') as token_file:
                token_file.write(creds.to_json())
            
            # Set secure file permissions
            token_path.chmod(0o600)
            print(f"Refreshed token saved to {token_path}")
            return
            
        except Exception as e:
            print(f"Error refreshing credentials: {e}")
            print("Will generate new credentials...")
            creds = None
    
    # Generate new credentials
    try:
        # Try to load client secrets from file
        if client_secrets_path.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), SCOPES)
        else:
            # Create from environment variables
            print("Client secrets file not found, using environment variables...")
            client_config = create_client_secrets_from_env()
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        
        print("Starting OAuth flow...")
        creds = flow.run_local_server(port=0)
        
        # Save new credentials
        with open(token_path, 'w') as token_file:
            token_file.write(creds.to_json())
        
        # Set secure file permissions
        token_path.chmod(0o600)
        print(f"New token saved to {token_path}")
        
    except Exception as e:
        print(f"Error generating credentials: {e}")
        raise

if __name__ == '__main__':
    main()
