# Security Setup Guide

## OAuth Credentials Setup

This application requires Google OAuth credentials for accessing Google Drive. Follow these steps to set up credentials securely:

### Option 1: Using Environment Variables (Recommended)

1. Set the following environment variables:
   ```bash
   export GOOGLE_CLIENT_ID="your-client-id"
   export GOOGLE_CLIENT_SECRET="your-client-secret"
   export GOOGLE_PROJECT_ID="your-project-id"
   ```

2. Run the token generation script:
   ```bash
   python generate_token.py
   ```

### Option 2: Using Client Secrets File

1. Download your OAuth client secrets JSON from Google Cloud Console
2. Set the path to your secrets file:
   ```bash
   export GOOGLE_CLIENT_SECRETS_PATH="/path/to/your/client_secrets.json"
   ```
3. Run the token generation script:
   ```bash
   python generate_token.py
   ```

### Obtaining Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Go to "Credentials" → "Create Credentials" → "OAuth client ID"
5. Choose "Desktop application"
6. Download the client secrets JSON file

### Security Notes

- **NEVER** commit `client_secrets.json` or `token.json` to version control
- Store credentials in a secure location outside your project directory
- Use environment variables for production deployments
- Regularly rotate your OAuth client secrets
- Set appropriate file permissions (600) on credential files

### File Locations

By default, credentials are stored in:
- **macOS/Linux**: `~/.config/microbiome-app/`
- **Windows**: `%APPDATA%\microbiome-app\`

You can override these locations with environment variables:
- `GOOGLE_CLIENT_SECRETS_PATH`: Path to client secrets file
- `GOOGLE_TOKEN_PATH`: Path to token file

### Troubleshooting

If you see authentication errors:
1. Check that your environment variables are set correctly
2. Verify your client secrets file exists and is readable
3. Ensure your OAuth client is configured for "Desktop application"
4. Check that redirect URIs include `http://localhost`

### Security Incident Response

If credentials are accidentally exposed:
1. Immediately revoke the OAuth client in Google Cloud Console
2. Generate new client credentials
3. Update your environment variables or secrets file
4. Re-run the token generation process
