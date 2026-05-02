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
    # Return status, output_ref, and a progress string
    output_ref = getattr(batch, "output_file_id", None) or getattr(batch, "error_file_id", None)
    
    counts = getattr(batch, "request_counts", None)
    progress = ""
    if counts:
        progress = f"({counts.completed + counts.failed}/{counts.total})"
    
    return batch.status, output_ref, progress

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
    
    counts = getattr(batch, "request_counts", None)
    progress = ""
    if counts:
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        total = done + counts.processing
        progress = f"({done}/{total})"
        
    return batch.processing_status, getattr(batch, "results_url", None), progress

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

def _is_ai_studio_batch(batch_id):
    """AI Studio batch IDs look like 'batches/xxx', Vertex IDs look like 'projects/.../batchPredictionJobs/xxx'."""
    return str(batch_id).startswith("batches/")


def _get_google_client(batch_id):
    """Return the right genai client based on the batch_id format."""
    from google import genai
    if _is_ai_studio_batch(batch_id):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY required to poll AI Studio batch jobs.")
        return genai.Client(api_key=api_key), "ai_studio"
    else:
        project = os.environ.get("GOOGLE_PROJECT_ID")
        location = os.environ.get("GOOGLE_LOCATION", "global")
        if "/locations/" in str(batch_id):
            location = batch_id.split("/locations/")[1].split("/")[0]
        client = genai.Client(vertexai=True, project=project, location=location)
        return client, "vertex"


def check_google_batch(batch_id):
    client, mode = _get_google_client(batch_id)
    batch_job = client.batches.get(name=batch_id)
    status = batch_job.state
    output_uri = ""

    if mode == "ai_studio":
        # AI Studio: output file referenced directly in the batch job object
        if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
            output_uri = batch_job.dest.file_name  # e.g. 'files/abc123'
    else:
        # Vertex AI: output in GCS
        if batch_job.dest:
            if hasattr(batch_job.dest, 'gcs_uri') and batch_job.dest.gcs_uri:
                output_uri = batch_job.dest.gcs_uri
            elif hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                output_uri = f"https://generativelanguage.googleapis.com/v1beta/{batch_job.dest.file_name}"

    return status, output_uri, ""

def download_google_batch(batch_id, output_uri, out_path):
    client, mode = _get_google_client(batch_id)

    if mode == "ai_studio":
        # --- AI Studio path ---
        # Re-fetch if we don't have the output URI yet
        if not output_uri:
            batch_job = client.batches.get(name=batch_id)
            if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                output_uri = batch_job.dest.file_name
        if not output_uri:
            print("  AI Studio batch: output file not available yet.")
            return False
        # Download via SDK
        print(f"  Downloading AI Studio batch output: {output_uri}")
        try:
            content = client.files.download(file=output_uri)
            with open(out_path, 'wb') as f:
                f.write(content)
            print(f"  Downloaded to {out_path}")
            return True
        except Exception as e:
            print(f"  SDK download failed ({e}), trying HTTP fallback...")
            api_key = os.environ.get("GOOGLE_API_KEY")
            try:
                dl_url = f"https://generativelanguage.googleapis.com/v1beta/{output_uri}?alt=media&key={api_key}"
                res = requests.get(dl_url, timeout=120)
                res.raise_for_status()
                with open(out_path, 'wb') as f:
                    f.write(res.content)
                print(f"  Downloaded (HTTP fallback) to {out_path}")
                return True
            except Exception as e2:
                print(f"  AI Studio download failed: SDK={e} | HTTP={e2}")
                return False

    else:
        # --- Vertex AI path (original) ---
        if not output_uri:
            batch_job = client.batches.get(name=batch_id)
            if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                output_uri = f"https://generativelanguage.googleapis.com/v1beta/{batch_job.dest.file_name}"
            elif batch_job.dest and hasattr(batch_job.dest, 'gcs_uri') and batch_job.dest.gcs_uri:
                output_uri = batch_job.dest.gcs_uri

        if output_uri:
            if "generativelanguage.googleapis.com" in output_uri:
                api_key = os.environ.get("GOOGLE_API_KEY")
                dl_url = f"{output_uri}?alt=media&key={api_key}"
                res = requests.get(dl_url)
                res.raise_for_status()
                with open(out_path, 'wb') as f:
                    f.write(res.content)
                return True
            elif output_uri.startswith("gs://"):
                print(f"  Detected GCS output folder/uri: {output_uri}")
                try:
                    from google.cloud import storage
                    project = os.environ.get("GOOGLE_PROJECT_ID")
                    gcs_client = storage.Client(project=project)
                    uri_no_prefix = output_uri.replace("gs://", "")
                    if "/" in uri_no_prefix:
                        bucket_name, prefix = uri_no_prefix.split("/", 1)
                    else:
                        bucket_name, prefix = uri_no_prefix, ""
                    bucket = gcs_client.bucket(bucket_name)
                    blobs = list(gcs_client.list_blobs(bucket, prefix=prefix))
                    result_blob = None
                    for b in blobs:
                        if b.name.endswith(".jsonl") and "input" not in b.name:
                            result_blob = b
                            break
                    if result_blob:
                        print(f"  Found result blob: {result_blob.name}")
                        result_blob.download_to_filename(out_path)
                        print(f"  Successfully downloaded to {out_path}")
                        return True
                    else:
                        print(f"  Could not find any .jsonl results in {output_uri}")
                        return False
                except Exception as e:
                    print(f"  Error downloading from GCS: {e}")
                    return False
        return False

def parse_openai_results(raw_path):
    """Returns (outputs dict, refusals set of custom_ids)."""
    outputs = {}
    refusals = set()
    with open(raw_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if "response" in data and "body" in data["response"] and "choices" in data["response"]["body"]:
                finish_reason = data["response"]["body"]["choices"][0].get("finish_reason", "")
                if finish_reason == "content_filter":
                    refusals.add(custom_id)
                    outputs[custom_id] = "ERROR: Refused by content filter"
                else:
                    outputs[custom_id] = data["response"]["body"]["choices"][0]["message"]["content"]
            else:
                outputs[custom_id] = "ERROR: " + str(data.get("error", "Unknown error"))
    return outputs, refusals

def parse_anthropic_results(raw_path):
    """Returns (outputs dict, refusals set of custom_ids)."""
    outputs = {}
    refusals = set()
    with open(raw_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if "result" in data and data.get("result", {}).get("type") == "succeeded":
                msg = data["result"].get("message", {})
                stop_reason = msg.get("stop_reason", "")
                content = msg.get("content", [])
                if stop_reason == "refusal":
                    refusals.add(custom_id)
                    outputs[custom_id] = "ERROR: Refused by model safety filter"
                elif content:
                    # Concatenate all thinking and text blocks
                    full_output = []
                    for block in content:
                        if block.get("type") == "text":
                            full_output.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            full_output.append(block.get("thinking", ""))
                    
                    if full_output:
                        outputs[custom_id] = "\n\n".join(full_output)
                    else:
                        outputs[custom_id] = "ERROR: No text or thinking blocks found in Anthropic response"
                else:
                    outputs[custom_id] = "ERROR: Empty content block in Anthropic response"
            else:
                outputs[custom_id] = "ERROR: " + str(data.get("result", {}).get("error", "Unknown error"))
    return outputs, refusals

def parse_google_results(raw_path):
    """Returns (outputs dict, refusals set of custom_ids)."""
    outputs = {}
    refusals = set()
    idx = 0
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            custom_id = data.get("request", {}).get("id") or data.get("id")
            
            msg_content = "ERROR: No response found"
            is_refusal = False
            if "response" in data and "candidates" in data["response"]:
                try:
                    candidate = data["response"]["candidates"][0]
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
                        is_refusal = True
                        msg_content = f"ERROR: Refused by safety filter ({finish_reason})"
                    else:
                        msg_content = candidate["content"]["parts"][0]["text"]
                except:
                    msg_content = "ERROR: Could not parse nested candidate response"
            elif "error" in data:
                msg_content = "ERROR: " + str(data.get("error", "Unknown error"))
            
            if custom_id:
                outputs[custom_id] = msg_content
                if is_refusal:
                    refusals.add(custom_id)
            # Fallback for when ID is missing (Vertex AI often omits it if not in specific format)
            outputs[idx] = msg_content
            idx += 1
    return outputs, refusals


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
            progress = ""
            if provider == "openai":
                status, output_ref, progress = check_openai_batch(batch_id)
            elif provider == "anthropic":
                status, output_ref, progress = check_anthropic_batch(batch_id)
            elif provider == "google":
                status, output_ref, progress = check_google_batch(batch_id)
                
            display_status = f"{status} {progress}".strip()
                
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
        
        print(f"[{idx}]   | {provider:<10} | {display_status:<15} | {exp:<20} | {ds}")
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
        timestamp = data.get('timestamp', 'unknown')
        raw_output_path = os.path.join(out_dir, f"batch_output_raw_{timestamp}_{batch_id.replace('/', '_')}.jsonl")
        
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
        refusal_ids = set()
        if provider == "openai":
            parsed_outputs, refusal_ids = parse_openai_results(raw_output_path)
        elif provider == "anthropic":
            parsed_outputs, refusal_ids = parse_anthropic_results(raw_output_path)
        elif provider == "google":
            parsed_outputs, refusal_ids = parse_google_results(raw_output_path)
            
        print(f"3. Grading results...")
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)
            
        # Reconstruct exactly like the sequential script output
        results = []
        stats = {"correct": 0, "total": 0, "failures": 0, "refusals": 0}
        for i, job in enumerate(jobs_data):
            custom_id = str(job['id']) + "_" + str(job.get('sample_idx', 0))
            
            # For Google, we try both ID and index because Vertex AI output formatting varies
            generated_text = parsed_outputs.get(custom_id)
            if generated_text is None and provider == "google":
                generated_text = parsed_outputs.get(i)
            
            if generated_text is None:
                generated_text = "ERROR: Missing from batch results"
            
            is_refusal = custom_id in refusal_ids

            try:
                extracted, is_correct = extract_and_grade(generated_text, job['ground_truth'])
            except Exception as e:
                extracted = f"ERROR: {str(e)}"
                is_correct = False
                
            result_entry = {
                "id": job['id'],
                "system_prompt": job.get('system_prompt', ''),
                "original": job.get('original', job.get('post_context_prompt', '')),
                "unmodified_original": job.get('unmodified_original', job.get('post_context_prompt', '')),
                "context_preview": job.get('context_preview', ''),
                "distractor_token_count": job.get('distractor_token_count', 0),
                "context_type": job.get('context_type', ''),
                "ground_truth": job['ground_truth'],
                "output": generated_text,
                "extracted": extracted,
                "correct": is_correct,
                "refusal": is_refusal
            }
            results.append(result_entry)
            
            stats["total"] += 1
            if is_correct: stats["correct"] += 1
            if is_refusal: stats["refusals"] += 1
            if not is_refusal and (extracted is None or (isinstance(extracted, str) and "ERROR" in extracted)):
                stats["failures"] += 1

        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        attempted = stats["total"] - stats["refusals"]
        acc_attempted = stats["correct"] / attempted if attempted > 0 else 0
        print(f"   Accuracy (all):      {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"   Accuracy (attempted): {acc_attempted:.2%} ({stats['correct']}/{attempted})")
        print(f"   Refusals: {stats['refusals']}")
        print(f"   Failures (non-refusal errors): {stats['failures']}")
        
        summary_data = {
            "accuracy": acc,
            "accuracy_attempted": acc_attempted,
            "correct": stats["correct"],
            "total": stats["total"],
            "attempted": attempted,
            "refusals": stats["refusals"],
            "failures": stats["failures"],
        }
        
        # Dynamically add all generation parameters recorded at launch (no fallbacks)
        internal_tracking_keys = {"batch_id", "provider", "google_mode", "model", "dataset", "experiment", "context_type", "context_size", "context_token_count", "timestamp", "jobs_file", "status", "metadata", "output_ref"}
        for k, v in data.items():
            if k not in internal_tracking_keys:
                summary_data[k] = v

        results.append({
            "summary": summary_data
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
        
        # Clean up: remove batch tracking and jobs files
        try:
            os.remove(tf)
            print(f"5. Removed tracking file: {tf}")
        except Exception as e:
            print(f"   Warning: Could not remove tracking file: {e}")
        try:
            os.remove(jobs_file)
            print(f"   Removed jobs file: {jobs_file}")
        except Exception as e:
            print(f"   Warning: Could not remove jobs file: {e}")
            
    print("\nAll done!")

if __name__ == "__main__":
    main()
