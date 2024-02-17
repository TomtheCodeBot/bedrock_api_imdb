# Generative AI Model Script with AWS Bedrock Integration

 This script is designed to call generative AI models using AWS Bedrock. Currently, it only supports sentiment analysis datasets.

# Prerequisites

 Before running the script, make sure you have the following prerequisites:

 - Required Python libraries installed (use `pip install -r requirements.txt`)
 - AWS credentials configured with appropriate permissions
 - AWS Bedrock API profile details

# Usage

 1. Install the required Python libraries:

     ```bash
      pip install -r requirements.txt
      ```

 2. Configure AWS credentials:

     ```bash
      export AWS_ACCESS_KEY_ID=your-access-key-id
      export AWS_SECRET_ACCESS_KEY=your-secret-access-key
      ```

 3. Run the script with desired parameters:

    ```bash
      python run.py --dataset your_dataset 
                        --subset your_subset 
                        --api-profile your_api_profile 
                        --default-region your_default_region 
                        --bedrock-assume-role your_assume_role 
                        --temperature 0 
                        --model-name your_model_name 
                        --top-p 0.1 
    ```
    
## Examples:

To run the script on the IMDB train dataset, using the Titan light model, run this following command:

    ```bash
      python run.py --dataset imdb 
                        --subset train 
                        --api-profile your_api_profile 
                        --default-region us-east-1 
                        --bedrock-assume-role your_assume_role 
                        --temperature 0 
                        --model-name amazon.titan-text-lite-v1:0:4k 
                        --top-p 0.1 
      ```

# Command-line Arguments

 - `--dataset`: Name of the dataset available in Hugging Face datasets (required).
 - `--subset`: Subset of the chosen dataset (default: 'train').
 - `--api-profile`: Bedrock API profile for making the AI model request (required).
 - `--default-region`: Default AWS region (default: 'us-east-1').
 - `--bedrock-assume-role`: ARN of the IAM role for Bedrock assume role (required).
 - `--temperature`: Temperature of the AI model request (default: 0).
 - `--model-name`: Name of the generative AI model (required).
 - `--top-p`: Top-p value for the AI model request (default: 0.1).
 - `--start-index`: Start index in the dataset for the process (default: None).
 - `--end-index`: End index in the dataset for the process (default: None).

# Output

 The script generates a CSV file containing the results, including the sentence, label, model response, and correctness. The output file is saved in the 'output' directory with a filename based on the current date, model name, and specified data range (e.g., "2022-02-17_your_model_name_full.csv").
