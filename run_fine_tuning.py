#!/usr/bin/env python3
"""
Run fine-tuning for VeriMedia toxicity classification using OpenAI API.
This script initiates and monitors the fine-tuning process.
"""

import os
import argparse
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Please add it to your .env file.")
    return OpenAI(api_key=api_key)

def create_fine_tuning_job(training_file_id, validation_file_id=None, model="gpt-3.5-turbo"):
    """Create a fine-tuning job with OpenAI API."""
    client = get_client()
    print(f"Creating fine-tuning job with model {model}...")
    
    job_params = {
        "training_file": training_file_id,
        "model": model,
        "suffix": "toxicity-classifier",
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
    
    try:
        response = client.fine_tuning.jobs.create(**job_params)
        print(f"Fine-tuning job created: {response.id}")
        print(f"Status: {response.status}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

def upload_file(file_path, purpose="fine-tune"):
    """Upload a file to OpenAI API."""
    client = get_client()
    print(f"Uploading {file_path}...")
    
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose=purpose
            )
        print(f"File uploaded: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def monitor_fine_tuning_job(job_id, check_interval=60):
    """Monitor the status of a fine-tuning job."""
    client = get_client()
    print(f"Monitoring fine-tuning job {job_id}...")
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"Status: {status}")
            
            if status == "succeeded":
                print(f"Fine-tuning completed successfully!")
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                print("Add this model ID to your .env file as FINE_TUNED_MODEL")
                return job.fine_tuned_model
            
            elif status == "failed":
                print(f"Fine-tuning failed: {job.error}")
                return None
            
            elif status in ["cancelled", "expired"]:
                print(f"Fine-tuning job {status}")
                return None
                
            # If still in progress, wait and check again
            print(f"Training metrics: {job.training_metrics if hasattr(job, 'training_metrics') else 'Not available'}")
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error monitoring fine-tuning job: {e}")
            time.sleep(check_interval)

def list_fine_tuned_models():
    """List all fine-tuned models."""
    try:
        client = get_client()
        models = client.models.list()
        
        fine_tuned_models = [model for model in models.data if "toxicity" in model.id]
        
        print("Available fine-tuned models:")
        for model in fine_tuned_models:
            print(f"- {model.id}")
            
    except Exception as e:
        print(f"Error listing models: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning for toxicity classification")
    parser.add_argument("--training-file", required=True, help="Path to training data JSONL file")
    parser.add_argument("--validation-file", help="Path to validation data JSONL file")
    parser.add_argument("--base-model", default="gpt-3.5-turbo", help="Base model to fine-tune")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--list-models", action="store_true", help="List available fine-tuned models")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_fine_tuned_models()
        return
    
    # Upload training file
    training_file_id = upload_file(args.training_file)
    if not training_file_id:
        print("Failed to upload training file. Exiting.")
        return
    
    # Upload validation file if provided
    validation_file_id = None
    if args.validation_file:
        validation_file_id = upload_file(args.validation_file)
        if not validation_file_id:
            print("Failed to upload validation file.")
    
    # Create fine-tuning job
    job_id = create_fine_tuning_job(training_file_id, validation_file_id, args.base_model)
    if not job_id:
        print("Failed to create fine-tuning job. Exiting.")
        return
    
    # Monitor fine-tuning job
    fine_tuned_model = monitor_fine_tuning_job(job_id, args.interval)
    
    if fine_tuned_model:
        # Save the model ID to finetune_model_id.txt for app.py compatibility
        with open("finetune_model_id.txt", "w") as f:
            f.write(fine_tuned_model)
            
        print("Model ID saved to finetune_model_id.txt")
        print("Add the following line to your .env file:")
        print(f"FINE_TUNED_MODEL={fine_tuned_model}")

if __name__ == "__main__":
    main() 