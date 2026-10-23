import os
import dotenv
from openai import OpenAI
import httpx

# Load environment variables
dotenv.load_dotenv()


def get_client():
    """Lazily initialize the OpenAI client when an API operation is requested."""
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key or openai_api_key in ('your_api_key_here', 'your_openai_api_key_here'):
        print("Error: Valid OPENAI_API_KEY not found in environment. Please add it to your .env file.")
        exit(1)
    return OpenAI(
        api_key=openai_api_key,
        http_client=httpx.Client()
    )


def create_fine_tuning_job():
    """
    Create and start a fine-tuning job for toxicity level classification
    """
    try:
        client = get_client()
        print("Starting fine-tuning process...")

        training_file_path = "data/fine_tuning/training.jsonl"
        validation_file_path = "data/fine_tuning/validation.jsonl"

        print(f"Uploading training file: {training_file_path}")
        with open(training_file_path, "rb") as f_train:
            training_file = client.files.create(
                file=f_train,
                purpose="fine-tune"
            )
        print(f"Training file uploaded with ID: {training_file.id}")

        print(f"Uploading validation file: {validation_file_path}")
        with open(validation_file_path, "rb") as f_val:
            validation_file = client.files.create(
                file=f_val,
                purpose="fine-tune"
            )
        print(f"Validation file uploaded with ID: {validation_file.id}")

        print("Creating fine-tuning job...")
        job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            validation_file=validation_file.id,
            model="gpt-3.5-turbo",
            suffix="toxicity-classifier",
            hyperparameters={
                "n_epochs": 3
            }
        )

        print(f"Fine-tuning job created with ID: {job.id}")
        print("The fine-tuning process has started. It may take some time to complete.")
        print("You can run 'python finetune.py --status' to check the status of your fine-tuning job.")

        with open("finetune_job_id.txt", "w", encoding="utf-8") as f:
            f.write(job.id)

        return job.id

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        return None


def check_fine_tuning_status(job_id=None):
    """
    Check the status of a fine-tuning job
    """
    try:
        client = get_client()
        if not job_id:
            try:
                with open("finetune_job_id.txt", "r", encoding="utf-8") as f:
                    job_id = f.read().strip()
            except FileNotFoundError:
                print("No job ID provided and no saved job ID found.")
                return None

        job = client.fine_tuning.jobs.retrieve(job_id)

        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Created at: {job.created_at}")
        print(f"Finished at: {job.finished_at if hasattr(job, 'finished_at') else 'Not finished yet'}")
        print(f"Model: {job.fine_tuned_model if hasattr(job, 'fine_tuned_model') else 'Not available yet'}")

        if job.status == "succeeded":
            print(f"\nFine-tuned model ID: {job.fine_tuned_model}")
            print("You can now use this model in your application.")

            with open("finetune_model_id.txt", "w", encoding="utf-8") as f:
                f.write(job.fine_tuned_model)

        return job

    except Exception as e:
        print(f"Error checking fine-tuning status: {str(e)}")
        return None


def list_fine_tuned_models():
    """
    List all fine-tuned models available to the user
    """
    try:
        client = get_client()
        print("Listing available fine-tuned models...")
        models = client.models.list()

        fine_tuned_models = [model for model in models.data if "ft" in model.id]

        if fine_tuned_models:
            print("\nYour fine-tuned models:")
            for model in fine_tuned_models:
                print(f"  - {model.id}")
        else:
            print("No fine-tuned models found.")

        return fine_tuned_models

    except Exception as e:
        print(f"Error listing fine-tuned models: {str(e)}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune an OpenAI model for toxicity classification")
    parser.add_argument("--start", action="store_true", help="Start a new fine-tuning job")
    parser.add_argument("--status", action="store_true", help="Check the status of the latest fine-tuning job")
    parser.add_argument("--job-id", type=str, help="Job ID to check status for")
    parser.add_argument("--list-models", action="store_true", help="List all fine-tuned models")

    args = parser.parse_args()

    if args.start:
        create_fine_tuning_job()
    elif args.status:
        check_fine_tuning_status(args.job_id)
    elif args.list_models:
        list_fine_tuned_models()
    else:
        parser.print_help()
