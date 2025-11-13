# Email_Compliance_GenAI

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

Upload an Excel file containing emails with columns: Date, From, To, Subject, Body.
The application will analyze emails for compliance risks using GPT-4o.

**Note:** Processed results are cached during the session. If you upload a different file, it will be re-analyzed. Downloading results won't trigger re-processing.
