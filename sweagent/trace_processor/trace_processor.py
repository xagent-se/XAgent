import json
from pathlib import Path
from tree_prunner.tracer_mod import read_tracer_output
from tree_prunner.repetition_pruning import RepetitionPruner, export_pruned_tree
from tree_prunner.extract_function_names import extract_function_names

def process_trace(trace_path: Path, output_dir: Path, trace_type: str, repo_name: str) -> None:
    try:
        func_scores, func_trees = read_tracer_output(str(trace_path))
        pruner = RepetitionPruner(min_subtree_size=1, min_frequency=2, important_keywords=[repo_name, "reproduc"])
        pruned_trees, _ = pruner.prune_trees(func_trees)
        pruned_txt = output_dir / f"{trace_type}_aggressive_pruned.txt"
        print(pruned_txt)
        fnames_txt = output_dir / f"{trace_type}_function_names.txt"
        fname_json = output_dir / f"{trace_type}_function_names.json"
        
        if not pruned_txt.exists():
            export_pruned_tree(pruned_trees, str(pruned_txt))
            pairs, counts, _, _ = extract_function_names(str(pruned_txt), [repo_name])
            fnames_txt = output_dir / f"{trace_type}_function_names.txt"
            if not fnames_txt.exists():
                with fnames_txt.open("w", encoding="utf-8") as f:
                    for func, file in sorted(pairs):
                        f.write(f"{func} -> {file} ({counts[(func, file)]})\n")
        
        fname_json = output_dir / f"{trace_type}_function_names.json"
        if not fname_json.exists():
            with fname_json.open("w", encoding="utf-8") as f:
                counts_json = {f"{func} -> {file}": counts[(func, file)] for func, file in pairs}
                json.dump(counts_json, f, indent=2)
            
        print(f"Processed {trace_path.name} in {output_dir.name}")
        return pruned_txt, fnames_txt, fname_json
    except Exception as e:
        print(f"Failed to process {trace_path}: {e}")
        return None, None, None

