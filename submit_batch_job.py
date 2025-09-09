import pandas as pd
from utils import to_request
import argparse
import json
from openai import OpenAI
import os
from dotenv import load_dotenv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='llama', help='Model ID to use')
    parser.add_argument('--activation_type', type=str, default=',mha', help='Activation type (residual, mlp, mha)')
    parser.add_argument('--chosen_layer', type=int, default=4, help='Which layer to intervene (int)')
    parser.add_argument('--scale', type=float, default=-10, help='Scale factor for probe vectors')
    # parser.add_argument('--run_num', type=int, default=0, help='Trial number for the experiment')
    
    args = parser.parse_args()
    model_id = args.model_id
    activation_type = args.activation_type
    chosen_layer = args.chosen_layer
    scale = args.scale
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    if activation_type == 'mha':
        prediction_path = f'predictions/{model_id}_answers_{chosen_layer}_{scale}_mha.csv'
    elif activation_type == 'mlp':
        prediction_path = f'predictions/{model_id}_answers_L[{chosen_layer}]_{scale}_mlp.csv'
    elif activation_type == 'residual':
         prediction_path = f'predictions/{model_id}_answers_L[{chosen_layer}]_{scale}_residual.csv'
    else:
        raise(f"typo {activation_type}")

    df = pd.read_csv(prediction_path)
    questions_test = df['question'].to_list()
    correct_answers_test = df['correct_answer'].to_list()
    initial_answer = df['initial_answer'].to_list()
    final_answer = df['final_answer'].to_list()

    print("Creating batch job for initial answers")
    req_desc = f'{model_id}_initial_{chosen_layer}_{scale}_{activation_type}'
    requests = [to_request(f'{req_desc}-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, initial_answer))]
    
    initial_answer_path = f"batch_job/{req_desc}.jsonl"
    with open(initial_answer_path, 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')


    batch_input_file = client.files.create(
        file=open(initial_answer_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": req_desc
        }
    )

    print("Creating batch job for final answers")
    req_desc = f'{model_id}_final_{chosen_layer}_{scale}_{activation_type}'
    requests = [to_request(f'{req_desc}-{i}', q, a, p) 
                for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, final_answer))]

    final_answer_path = f"batch_job/{req_desc}.jsonl"
    with open(final_answer_path, 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    batch_input_file = client.files.create(
        file=open(final_answer_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": req_desc
        }
    )
