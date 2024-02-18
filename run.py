import json
import os
import sys
import csv

import boto3
import botocore
import datetime
from utils import bedrock, print_ww,correct_check
from time import sleep
import pandas as pd
from tqdm import tqdm
import argparse
import datasets
from langchain_community.chat_models.anthropic import (
    convert_messages_to_prompt_anthropic,
)
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama
from langchain.schema import HumanMessage

prompt_data = """For the paragraph: '{}', is the sentiment in this paragraph positive or negative? Answer in one word.\n\n"""
def create_arg_parser():
    parser = argparse.ArgumentParser(description="Script to call generative AI models in AWS Bedrock.")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset available in Hugging Face datasets."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default='train',
        help="Subset of the chosen dataset. (default to 'train')."
    )

    parser.add_argument(
        "--api-profile",
        type=str,
        required=True,
        help="Bedrock API profile for making the AI model request."
    )
    
    parser.add_argument(
        "--default-region",
        type=str,
        default="us-east-1",
        help="Default AWS region. (default to 'us-east-1')."
    )

    parser.add_argument(
        "--bedrock-assume-role",
        type=str,
        required=True,
        help="ARN of the IAM role for Bedrock assume role."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature of the AI model request."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the generative AI model."
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.1,
        help="Top-p value for the AI model request."
    )


    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index in the dataset for the process (default is None)."
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index in the dataset for the process (default is None)."
    )

    return parser
def generate_response(prompt, args, boto3_bedrock):
    # model: str, prompt: str, max_token: maximm number of generated tokens
    message = HumanMessage(content =prompt)
    if "titan" in args.model_name:
        prompt = convert_messages_to_prompt_anthropic(
            messages=[message],
            human_prompt="\n\nUser:",
            ai_prompt="\n\nBot:",
        )
        body = json.dumps({
            "inputText": prompt, 
            "textGenerationConfig":{
                "maxTokenCount":10,
                "stopSequences":[],
                "temperature":args.temperature,
                "topP":args.top_p
                }
            }) 
    elif "cohere" in args.model_name:
        body = json.dumps({
            "prompt": prompt, 
            "max_tokens":10,
            "temperature":args.temperature,
            "p":args.top_p
            }
        ) 
    elif "meta" in args.model_name:
        prompt = convert_messages_to_prompt_llama(messages=[message])
        body = json.dumps({
            "prompt": prompt, 
            "max_gen_len":10,
            "temperature":args.temperature,
            "top_p":args.top_p
            }
        ) 
    else:
        raise Exception("model is yet to be implemented.")
    modelId = args.model_name # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'
    outputText = "\n"
    try:

        response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        
        response_body = json.loads(response.get('body').read())
        if "titan" in args.model_name:
            outputText = response_body.get('results')[0].get('outputText')
        elif "cohere" in args.model_name:
            outputText = response_body.get('generations')[0].get('text')
        elif "meta" in args.model_name:
            outputText = response_body.get('generation')
        return outputText
    except botocore.exceptions.ClientError as error:
        
        if error.response['Error']['Code'] == 'AccessDeniedException':
            print(f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                    \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                    \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
            
        raise error
def api_generate_resp( prompt, args,boto3_bedrock, max_retries=5):
    if max_retries == 0:
        print(prompt)
        now = datetime.datetime.now()
        file_name = now.strftime("%Y%m%d-%H%M%S%f")
        print(file_name)
        with open(f"error/{file_name}.txt","w") as f:
            f.write(prompt)
        return None
    else:
        try:
            resp = generate_response( prompt, args,boto3_bedrock)
            return resp
        except Exception as e:
            print(f"[ERROR]:{e} - Attempt to run left: {str(max_retries)}")
            sleep(5)
            return api_generate_resp( prompt, args,boto3_bedrock, max_retries-1)

def run(name_output,data,args,boto3_bedrock):
    with open(f'{name_output}.csv', mode="a", newline="") as f:
        # Create a writer object
        writer = csv.writer(f,delimiter="\t")
        writer.writerow(["sentence", "label", "response", "correct"])

        labels = [False,True]
        for i in tqdm(range(data.shape[0])):
            prompt = prompt_data.format(data['text'][i])
            row = [data['text'][i]]
            row.append(data['label'][i].item())
            label =labels[data['label'][i].item()] 
            response = api_generate_resp(prompt,args,boto3_bedrock)
            correct = correct_check.check_correct(response,label)
            row.append(response)
            row.append(correct)
            # Write the data to the CSV file
            writer.writerow(row)




if __name__ == "__main__":

    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    # Your logic for calling the generative AI models in AWS Bedrock goes here
    
    print(f"[INFO]:Loading datasets - {args.dataset}")
    dataset = pd.DataFrame(datasets.load_dataset(args.dataset,split = args.subset))

    print("[INFO]:Setting up bedrock client")

    os.environ["AWS_DEFAULT_REGION"] = args.default_region
    os.environ["AWS_PROFILE"] = args.api_profile
    os.environ["BEDROCK_ASSUME_ROLE"] = args.bedrock_assume_role

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    model_name = args.model_name  # Replace with your actual model name
    
    if (args.start_index is not None) !=  (args.end_index is not None):
        raise Exception("[ERROR]:Both start index and end index needs to be inserted in order for to accurately generate results. To run the full data,leave BOTH parameters empty.")
    elif args.start_index is not None:
        print(f"[INFO]: Generating results for {args.subset} data between {args.start_index} - {args.end_index}.")
        specified_length = f"{(str(args.start_index))}-{str(args.end_index)}"
        dataset = dataset.iloc[args.start_index:args.end_index]
    else:
        print(f"[INFO]: Generating results entire {args.subset} data.")
        specified_length = "full"
    name_output = f"{date_str}_{model_name}_{args.subset}_{specified_length}.csv"
    output_destination = f"output/{name_output}"

    print(f"[INFO]: Generating responds")
    print(f"[INFO]: Output destintation: {output_destination}")
    run(output_destination,dataset,args,boto3_bedrock)