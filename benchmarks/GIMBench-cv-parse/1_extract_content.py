import json

from pathlib import Path


def extract_content():
    base_dir = Path(__file__).parent
    raw_cvs_dir = base_dir / "data" / "raw_cvs"
    output_dir = base_dir / "data" / "mineru_output"
    result_file = base_dir / "data" / "extracted_content.jsonl"

    if not output_dir.exists():
        print(f"Error: Mineru output directory not found at {output_dir}")
        print("Please run the mineru command first.")
        return

    pdf_files = list(raw_cvs_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} source PDF files.")

    results = []

    for i, pdf_file in enumerate(pdf_files):
        # Mineru extracted structure: output_dir / {filename_stem} / {filename_stem}.md
        # subfolder name logic might vary if there are spaces, but typically it is the stem.
        stem = pdf_file.stem
        subfolder = output_dir / stem

        # If folder doesn't exist, try one with cleaned name perhaps?
        # But let's assume standard mineru behavior.
        if not subfolder.exists():
            print(f"[{i + 1}/{len(pdf_files)}] Missing output folder for: {pdf_file.name} (checked {subfolder.name})")
            continue

        # Find markdown file
        # Priority: {stem}.md inside that folder
        target_md_path = subfolder / f"{stem}.md"

        if not target_md_path.exists():
            # Fallback: search any md file in that subfolder recursively (e.g. inside hybrid_auto)
            md_files = list(subfolder.rglob("*.md"))
            if not md_files:
                print(f"[{i + 1}/{len(pdf_files)}] No markdown found in {subfolder} for {pdf_file.name}")
                continue
            # Prefer the one matching the stem if multiple
            target_md_path = md_files[0]
            for md in md_files:
                if md.stem == stem:
                    target_md_path = md
                    break

        try:
            content = target_md_path.read_text(encoding="utf-8")

            # Simple check to skip empty or extremely short content
            if len(content.strip()) < 10:
                print(f"[{i + 1}/{len(pdf_files)}] Warning: Empty/short content for {pdf_file.name}")

            results.append(
                {"original_filename": pdf_file.name, "original_path": str(pdf_file), "extracted_text": content}
            )

        except Exception as e:
            print(f"Error reading {target_md_path}: {e}")

    # Write results
    with open(result_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in results)

    print("\nProcessing complete.")
    print(f"Successfully loaded {len(results)}/{len(pdf_files)} documents.")
    print(f"Saved to {result_file}")


if __name__ == "__main__":
    extract_content()
