import argparse
import csv
import json
from pathlib import Path


def _load_records(input_path: Path) -> list[dict]:
	"""
	Load records from either:
	- JSON Lines (one JSON object per line), or
	- a JSON array of objects.
	"""
	text = input_path.read_text(encoding="utf-8").strip()
	if not text:
		return []

	# Try full-document JSON first (supports array or object-of-objects).
	try:
		data = json.loads(text)
		if isinstance(data, list):
			records = [r for r in data if isinstance(r, dict)]
			if len(records) != len(data):
				raise ValueError("All elements in JSON array must be objects.")
			return records
		if isinstance(data, dict):
			# Accept either one record object or a mapping like
			# {"dataset1": {...}, "dataset2": {...}}
			if all(isinstance(v, dict) for v in data.values()):
				return list(data.values())
			return [data]
	except json.JSONDecodeError:
		# Fallback to JSON Lines mode below.
		pass

	# JSON Lines mode
	records = []
	for idx, line in enumerate(text.splitlines(), start=1):
		line = line.strip()
		if not line:
			continue
		obj = json.loads(line)
		if not isinstance(obj, dict):
			raise ValueError(f"Line {idx} is not a JSON object.")
		records.append(obj)
	return records


def _to_cell(value):
	"""Convert nested values to compact JSON strings for CSV cells."""
	def _round_nested(obj):
		if isinstance(obj, bool):
			return obj
		if isinstance(obj, float):
			return round(obj, 2)
		if isinstance(obj, dict):
			return {k: _round_nested(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [_round_nested(v) for v in obj]
		return obj

	if isinstance(value, bool):
		return value
	if isinstance(value, float):
		return f"{value:.2f}"
	if isinstance(value, (dict, list)):
		rounded = _round_nested(value)
		return json.dumps(rounded, separators=(",", ":"), ensure_ascii=False)
	return value


def _compute_reported_gap(record: dict):
	"""Compute percentage gap using reported_ub and lb: ((reported_ub - lb) / lb) * 100."""
	try:
		reported_ub = float(record.get("reported_ub"))
		lb = float(record.get("lb"))
		if lb == 0:
			return ""
		return (reported_ub - lb) / lb * 100
	except (TypeError, ValueError):
		return ""


def convert_json_to_csv(input_file: str = "all_results.json", output_file: str = "all_results.csv") -> int:
	input_path = Path(input_file)
	output_path = Path(output_file)

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	records = _load_records(input_path)
	if not records:
		with output_path.open("w", newline="", encoding="utf-8") as f:
			f.write("")
		return 0

	# Column order: first-seen key order across records.
	columns = []
	seen = set()
	for record in records:
		for key in record.keys():
			if key not in seen:
				seen.add(key)
				columns.append(key)

	computed_col = "gap_reported_ub_lb"
	if computed_col not in seen:
		if "gap" in columns:
			gap_idx = columns.index("gap")
			columns.insert(gap_idx + 1, computed_col)
		else:
			columns.append(computed_col)

	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=columns)
		writer.writeheader()
		for record in records:
			row_source = dict(record)
			row_source[computed_col] = _compute_reported_gap(record)
			row = {col: _to_cell(row_source.get(col, "")) for col in columns}
			writer.writerow(row)

	return len(records)


def main():
	parser = argparse.ArgumentParser(description="Convert all_results JSON/JSONL file to CSV.")
	parser.add_argument("-i", "--input", default="all_results.json", help="Input JSON/JSONL file path")
	parser.add_argument("-o", "--output", default="all_results.csv", help="Output CSV file path")
	args = parser.parse_args()

	count = convert_json_to_csv(args.input, args.output)
	print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
	main()
