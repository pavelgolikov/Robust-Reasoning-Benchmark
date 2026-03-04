#!/usr/bin/env python3
"""
Interactive script to poll status of async batch API jobs and download/grade them once complete.

Usage:
    python poll_and_grade_batches.py
"""

import os
import json
import glob
import time
import argparse
import requests
from util import extract_and_grade

# We need the provider logic to check status and download
from api_utils import BATCH_PROVIDER_REGISTRY

def check_openai_batch(batch_id):
    from openai import OpenAI
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    return batch.status, getattr(batch, "output_file_id", None)

def download_openai_batch(file_id, out_path):
    from openai import OpenAI
    client = OpenAI()
    content = client.files.content(file_id).text
    with open(out_path, 'w') as f:
        f.write(content)
    return True

def check_anthropic_batch(batch_id):
    from anthropic import Anthropic
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    return batch.processing_status, getattr(batch.results_url, None) if hasattr(batch, 'results_url') else None

def download_anthropic_batch(batch_id, out_path):
    from anthropic import Anthropic
    client = Anthropic()
    # Anthropic SDK handles result stream
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(result.to_dict())
    
    with open(out_path, 'w') as f:
        # Save as jsonl to match pattern
        for r in results:
            f.write(json.dumps(r) + "\n")
    return True

def check_google_batch(batch_id):
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # In the new SDK, batch_id is the full resource name
    batch_job = client.batches.get(name=batch_id)
    
    status = batch_job.state
    output_uri = ""
    # Look for output file in dest
    if batch_job.dest:
        if hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
            output_uri = f"https://generativelanguage.googleapis.com/v1beta/{batch_job.dest.file_name}"
        elif hasattr(batch_job.dest, 'gcs_uri') and batch_job.dest.gcs_uri:
            output_uri = batch_job.dest.gcs_uri
            
    return status, output_uri

def download_google_batch(batch_id, output_uri, out_path):
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not output_uri:
        client = genai.Client(api_key=api_key)
        batch_job = client.batches.get(name=batch_id)
        if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
            output_uri = f"https://generativelanguage.googleapis.com/v1beta/{batch_job.dest.file_name}"

    if output_uri:
        if "generativelanguage.googleapis.com" in output_uri:
             dl_url = f"{output_uri}?alt=media&key={api_key}"
             res = requests.get(dl_url)
             res.raise_for_status()
             with open(out_path, 'wb') as f:
                 f.write(res.content)
             return True
        elif output_uri.startswith("gs://"):
             print(f"  Warning: Output is on GCS ({output_uri}). Please download manually or install gcloud.")
             return False
    return False

def parse_openai_results(raw_path):
    outputs = {}
    with open(raw_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if "response" in data and "body" in data["response"] and "choices" in data["response"]["body"]:
                msg_content = data["response"]["body"]["choices"][0]["message"]["content"]
                outputs[custom_id] = msg_content
            else:
                outputs[custom_id] = "ERROR: " + str(data.get("error", "Unknown error"))
    return outputs

def parse_anthropic_results(raw_path):
    outputs = {}
    with open(raw_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if "result" in data and data["result"]["type"] == "succeeded":
                outputs[custom_id] = data["result"]["message"]["content"][0]["text"]
            else:
                outputs[custom_id] = "ERROR: " + str(data.get("result", {}).get("error", "Unknown error"))
    return outputs

def parse_google_results(raw_path):
    outputs = {}
    with open(raw_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("request", {}).get("id") or data.get("id")
            if "response" in data and "candidates" in data["response"]:
                try:
                    msg_content = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    outputs[custom_id] = msg_content
                except:
                    outputs[custom_id] = "ERROR: Could not parse nested candidate response"
            else:
                outputs[custom_id] = "ERROR: " + str(data.get("error", "Unknown error"))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Poll and grade async batch API jobs")
    parser.add_argument("--dir", type=str, default=".", help="Base directory to search for batch_tracking JSON files")
    args = parser.parse_args()

    # Find all tracking files
    base_dir = args.dir
    tracking_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("batch_tracking_") and file.endswith(".json"):
                tracking_files.append(os.path.join(root, file))

    if not tracking_files:
        print(f"No batch tracking files found in {base_dir}")
        return

    # Filter out already completed ones
    active_batches = []
    completed_batches = []
    
    for tf in tracking_files:
        with open(tf, 'r') as f:
            data = json.load(f)
            
        if data.get("status") == "COMPLETED_AND_GRADED":
            continue
            
        active_batches.append((tf, data))

    if not active_batches:
        print("All found batches are already graded and completed!")
        return

    print(f"Found {len(active_batches)} tracked batches needing attention.\n")
    
    # 1. Poll Status
    print(f"{'INDEX':<5} | {'PROVIDER':<10} | {'STATUS':<15} | {'EXPERIMENT':<20} | {'DATASET'}")
    print("-" * 75)
    
    poll_results = []
    
    for idx, (tf, data) in enumerate(active_batches):
        provider = data["provider"]
        batch_id = data["batch_id"]
        status = data.get("status", "UNKNOWN")
        output_ref = None
        
        try:
            if provider == "openai":
                status, output_ref = check_openai_batch(batch_id)
            elif provider == "anthropic":
                status, output_ref = check_anthropic_batch(batch_id)
            elif provider == "google":
                status, output_ref = check_google_batch(batch_id)
                
            # Update tracking file with current status
            data["status"] = status
            if output_ref:
                data["output_ref"] = output_ref
            with open(tf, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            status = f"ERROR: {str(e)[:20]}"
            
        exp = data.get("experiment", "unknown")
        ds = data.get("dataset", "unknown").split('/')[-1]
        
        print(f"[{idx}]   | {provider:<10} | {status:<15} | {exp:<20} | {ds}")
        poll_results.append((tf, data, status, output_ref))

    # Identify downloadable ones
    # OpenAI: "completed", Anthropic: "ended", Google: "SUCCEEDED" / "JOB_STATE_SUCCEEDED"
    downloadable = []
    for idx, (tf, data, status, output_ref) in enumerate(poll_results):
        s_low = str(status).lower()
        if "completed" in s_low or "ended" in s_low or "succeeded" in s_low:
            downloadable.append(idx)

    if not downloadable:
        print("\nNo batches are fully completed yet. Check back later.")
        return

    print(f"\nBatches {downloadable} are marked as completed on the provider side.")
    ans = input("Do you want to download and grade them now? (Yes/No): ")
    if ans.strip().lower() not in ["yes", "y"]:
        print("Exiting.")
        return

    # 2. Download and Grade
    for idx in downloadable:
        tf, data, status, output_ref = poll_results[idx]
        provider = data["provider"]
        batch_id = data["batch_id"]
        jobs_file = data["jobs_file"]
        
        print(f"\n=== Processing Batch [{idx}] ({provider} - {data['experiment']}) ===")
        
        out_dir = os.path.dirname(tf)
        raw_output_path = os.path.join(out_dir, f"batch_output_raw_{batch_id.replace('/', '_')}.jsonl")
        
        print(f"1. Downloading raw output from {provider}...")
        try:
            if provider == "openai":
                if not output_ref:
                    print(f"  Warning: No output file ID for {batch_id}")
                    continue
                download_openai_batch(output_ref, raw_output_path)
            elif provider == "anthropic":
                download_anthropic_batch(batch_id, raw_output_path)
            elif provider == "google":
                download_google_batch(batch_id, output_ref, raw_output_path)
        except Exception as e:
            print(f"  Error downloading batch {batch_id}: {e}")
            continue
            
        print(f"2. Parsing raw outputs...")
        parsed_outputs = {}
        if provider == "openai":
            parsed_outputs = parse_openai_results(raw_output_path)
        elif provider == "anthropic":
            parsed_outputs = parse_anthropic_results(raw_output_path)
        elif provider == "google":
            parsed_outputs = parse_google_results(raw_output_path)
            
        print(f"3. Grading results...")
        with open(jobs_file, 'r') as f:
            jobs = json.load(f)
            
        # Reconstruct exactly like the sequential script output
        results = []
        stats = {"correct": 0, "total": 0, "failures": 0}
        
        for job in jobs:
            custom_id = str(job['id']) + "_" + str(job.get('sample_idx', 0))
            generated_text = parsed_outputs.get(custom_id, "ERROR: Missing from batch results")
            
            try:
                extracted, is_correct = extract_and_grade(generated_text, job['ground_truth'])
            except Exception as e:
                extracted = f"ERROR: {str(e)}"
                is_correct = False
                
            result_entry = {
                "id": job['id'],
                "system_prompt": job['system_prompt'],
                "original": job['original'],
                "unmodified_original": job['unmodified_original'],
                "ground_truth": job['ground_truth'],
                "output": generated_text,
                "extracted": extracted,
                "correct": is_correct
            }
            results.append(result_entry)
            
            stats["total"] += 1
            if is_correct: stats["correct"] += 1
            if extracted is None or (isinstance(extracted, str) and "ERROR" in extracted):
                stats["failures"] += 1

        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"   Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"   Failures: {stats['failures']}")
        
        results.append({
            "summary": {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
                "failures": stats["failures"]
            }
        })
        
        # Save final
        base_name = os.path.basename(jobs_file).replace("jobs_", "").replace(".json", "")
        safe_model = data['model'].replace('/', '_').replace(' ', '_')
        safe_dataset = data['dataset'].replace('/', '_')
        exp = data['experiment']
        timestamp = data['timestamp']
        
        run_id = f"{safe_model}_{safe_dataset}_{exp}_{timestamp}"
        final_json_file = os.path.join(out_dir, f"{run_id}.json")
        
        with open(final_json_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"4. Saved final graded results to: {final_json_file}")
        
        # Mark tracking as totally done
        data["status"] = "COMPLETED_AND_GRADED"
        with open(tf, 'w') as f:
            json.dump(data, f, indent=2)
            
    print("\nAll done!")

if __name__ == "__main__":
    main()
