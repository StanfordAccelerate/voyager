import os
import sys
import requests
import pandas as pd
import zipfile
import io

# GitHub API setup
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]
PR_NUMBER = os.environ["PR_NUMBER"]
CURRENT_RUN_ID = os.environ["CURRENT_RUN_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
BASE_URL = f"https://api.github.com/repos/{REPO}"

def get_latest_main_run_id():
    """Find the latest successful run on the main branch."""
    url = f"{BASE_URL}/actions/runs?branch=main&status=success&per_page=1"
    response = requests.get(url, headers=HEADERS).json()
    if not response.get("workflow_runs"):
        return None
    return response["workflow_runs"][0]["id"]

def download_and_parse_artifacts(run_id):
    """Download artifacts from a run, extract test_results.pkl, and build a DataFrame."""
    url = f"{BASE_URL}/actions/runs/{run_id}/artifacts"
    response = requests.get(url, headers=HEADERS).json()
    
    df = pd.DataFrame()
    
    for artifact in response.get("artifacts", []):
        if not artifact["name"].startswith("regression-results-"):
            continue
            
        # Extract metadata from artifact name (e.g., regression-results-E4M3-16x16)
        parts = artifact["name"].replace("regression-results-", "").split("-")
        datatype = parts[0]
        rows, cols = map(int, parts[1].split("x"))
        
        # Download the artifact zip
        dl_url = artifact["archive_download_url"]
        dl_response = requests.get(dl_url, headers=HEADERS, allow_redirects=True)
        
        if dl_response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(dl_response.content)) as z:
                # We expect test_results.pkl to be at the root of the artifact zip based on upload paths
                with z.open("test_results.pkl") as f:
                    job_df = pd.read_pickle(f)
                    job_df["datatype"] = datatype
                    job_df["rows"] = rows
                    job_df["cols"] = cols
                    # Defaults for missing buffers in artifact naming convention
                    job_df["input_buffer_size"] = 1024
                    job_df["weight_buffer_size"] = 1024
                    job_df["output_buffer_size"] = 1024
                    df = pd.concat([df, job_df])
    return df

def main():
    main_run_id = get_latest_main_run_id()
    if not main_run_id:
        print("No successful main run found to compare against.")
        sys.exit(0)

    main_df = download_and_parse_artifacts(main_run_id)
    pr_df = download_and_parse_artifacts(CURRENT_RUN_ID)

    if pr_df.empty:
        print("No regression results found in the current PR run.")
        sys.exit(0)

    categories = ["datatype", "rows", "cols", "input_buffer_size", "weight_buffer_size", "output_buffer_size", "Model"]

    def compute_grouped_runtime(df):
        if {"Count", "L2 Tiles", "Actual Tiles"}.issubset(df.columns):
            df = df.assign(weighted_runtime=(df["Runtime"] * df["Count"] * df["L2 Tiles"] / df["Actual Tiles"]))
        elif {"Count", "L2 Tiles"}.issubset(df.columns):
            df = df.assign(weighted_runtime=(df["Runtime"] * df["Count"] * df["L2 Tiles"]))
        else:
            return df.groupby(categories)["Runtime"].sum()
        return df.groupby(categories, as_index=True)["weighted_runtime"].sum()

    main_grouped = compute_grouped_runtime(main_df)
    pr_grouped = compute_grouped_runtime(pr_df)

    main_grouped.name = "runtime_main"
    pr_grouped.name = "runtime_pr"

    merged = pd.merge(main_grouped, pr_grouped, left_index=True, right_index=True, suffixes=("_main", "_pr"), how="outer")
    merged["runtime_diff"] = merged["runtime_main"] - merged["runtime_pr"]

    negative_diff = merged[merged["runtime_diff"] < 0]
    
    comment = ""
    passed = True
    if not negative_diff.empty:
        comment += "⚠️⚠️ The following configurations have a **longer runtime** on this PR:\n\n```text\n"
        comment += negative_diff.to_string() + "\n```\n"
        passed = False
    else:
        comment += "✅ The runtime of all configurations is the same or shorter on this PR branch.\n"

    comment += "\n### Performance Comparison\n```text\n" + merged.to_string() + "\n```\n"
    
    print(comment)

    # Post comment to PR
    comments_url = f"{BASE_URL}/issues/{PR_NUMBER}/comments"
    requests.post(comments_url, headers=HEADERS, json={"body": comment})

    # Fail the GitHub Action step if there are performance regressions
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
