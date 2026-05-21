# AI Campaign Optimizer — Local setup

Set up your Google Gemini API key (do NOT commit this key).

1. Copy `.env.example` to `.env` inside the `campaign_mvp` folder and replace the placeholder:

```powershell
cd "c:\Users\madan\OneDrive\Desktop\ai Campaign Optimizer\campaign_mvp"
copy .env.example .env
(Get-Content .env) -replace 'REPLACE_WITH_YOUR_GEMINI_API_KEY','YOUR_REAL_KEY' | Set-Content .env
```

2. Or set for current PowerShell session:

```powershell
$env:GEMINI_API_KEY = 'YOUR_REAL_KEY'
```

3. Verify the env var is available in PowerShell:

```powershell
echo $env:GEMINI_API_KEY
```

4. Run the app:

```powershell
python -m streamlit run app.py
```

If you still see "API key not valid" from the Gemini API, confirm you copied the correct API key from the Google Cloud Console and that the key has the proper access enabled for `generativelanguage.googleapis.com`.
