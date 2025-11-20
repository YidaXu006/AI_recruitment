# AI_recruitment
Deploy the AI-powered automatic resume scoring workflow locally.

## work_flow_1.py
This script implements an AI-powered resume scoring workflow. It processes resumes in various formats (DOCX, PDF), extracts text content, and scores the resumes based on predefined criteria using your local AI model.

## AI_models
This project uses the deepseek-r1:7b model for resume scoring.You should download ollama and pull the model deepseek-r1:7b before running the script.You can also download more powerful models if you have a good GPU.

## .env
You should set the following environment variables in the .env file:
- JD_FILE_PATH: The path to the job description file (DOCX format).
- RESUME_FILE_PATH: The path to the resume file (DOCX or PDF format).
- MODEL_NAME: The name of the local AI model to use for scoring (default is deepseek-r1:7b).