import json
import os

from pathlib import Path

import json_repair

from openai import OpenAI
from pydantic import ValidationError
from schemas import CVData, cv_data_schema


def extract_fields():
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "extracted_content.jsonl"
    output_file = base_dir / "data" / "parsed_fields.jsonl"

    # Check API Key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if not input_file.exists():
        print(f"Input file {input_file} not found. Run 1_extract_content.py first.")
        return

    results = []

    with open(input_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Found {len(records)} records to process.")

    for i, record in enumerate(records[:4]):
        print(f"Processing [{i + 1}/{len(records)}]: {record['original_filename']}")

        cv_text = record.get("extracted_text", "")
        if not cv_text:
            print("  Warning: No text content.")
            continue
        prompt = (
            "You are an expert CV parser. Extract the user information from the following CV text into a structured JSON object.\n\n"
            "## CV Text\n"
            f"{cv_text}"
            "## Note\n"
            "- "
        )

        try:
            response = client.chat.completions.create(
                model="openai/gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "cv_extraction", "strict": True, "schema": cv_data_schema},
                },
                temperature=0.1,
            )

            content = response.choices[0].message.content
            if not content:
                print("  Error: Empty response")
                continue

            # Parse and Validate
            try:
                data_dict = json_repair.loads(content)
                cv_data = CVData.model_validate(data_dict)

                # Store structured data
                result_record = record.copy()
                result_record["parsed_data"] = cv_data.model_dump()
                results.append(result_record)
                print("  Success")

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"  Error parsing JSON: {e}")
                # Save raw response for debugging?
                result_record = record.copy()
                result_record["parsed_data_raw"] = content
                result_record["error"] = str(e)
                results.append(result_record)

        except Exception as e:
            print(f"  API Error: {e}")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Processing complete. Saved to {output_file}")


if __name__ == "__main__":
    extract_fields()
