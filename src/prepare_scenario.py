#!/usr/bin/env python3
"""
CLI to prepare scenario assets:
 1. Copy database KB file (e.g., archeology_scan_kb.jsonl) to scenario/knowledge.jsonl
 2. Append safety_rules to knowledge.jsonl with auto-incremented IDs
 3. Fork reference DB, run gold_query → expected_snapshot.json

Usage:
  python src/prepare_scenario.py --scenario scenarios/business_and_integrity_rules/massive_delete/1_archeology

The script:
  - Copies livesqlbench-base-full-v1/<db>/<db>_kb.jsonl to <scenario>/knowledge.jsonl
  - Appends safety_rules from scenario.json with auto-assigned IDs
  - Runs gold_query to create expected_snapshot.json
"""

import sys
import json
import argparse
import shutil
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.kb_sources import (
    load_base_column_meanings,
    resolve_column_meanings_file,
    resolve_kb_file,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare scenario assets (knowledge + snapshot)")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario path relative to repo root (e.g., scenarios/business_and_integrity_rules/massive_delete/1_archeology)",
    )
    args = parser.parse_args()

    # Normalize scenario path
    scen_input = args.scenario.strip().rstrip("/\\")
    scen_dir = Path(project_root) / scen_input

    if not scen_dir.exists():
        print(json.dumps({"status": "error", "error": f"Scenario directory not found: {scen_dir}"}, indent=2))
        return 1

    try:
        # Step 1: Load scenario.json
        scenario_file = scen_dir / "scenario.json"
        if not scenario_file.exists():
            print(json.dumps({"status": "error", "error": f"scenario.json not found in {scen_dir}"}, indent=2))
            return 1

        with open(scenario_file) as f:
            scenario_data = json.load(f)

        reference_db = scenario_data.get("reference_database")
        if not reference_db:
            print(json.dumps({"status": "error", "error": "reference_database not specified in scenario.json"}, indent=2))
            return 1

        # Step 2: Copy KB + semantics (if needed)
        knowledge_file = scen_dir / "knowledge.jsonl"
        column_meanings_file = scen_dir / "column_meanings.json"

        kb_source_file = resolve_kb_file(reference_db, project_root)
        column_meanings_source = resolve_column_meanings_file(reference_db, project_root)

        if not knowledge_file.exists() or knowledge_file.stat().st_size == 0:
            if kb_source_file and kb_source_file.exists():
                shutil.copy2(kb_source_file, knowledge_file)
                print(f"Copied {kb_source_file} to {knowledge_file}")
            else:
                with open(knowledge_file, 'w') as f:
                    pass
                if kb_source_file is None:
                    print(f"Warning: No KB mapping for '{reference_db}', created empty {knowledge_file}")
                else:
                    print(f"Warning: KB file not found ({kb_source_file}), created empty {knowledge_file}")
        else:
            print(f"Kept existing {knowledge_file}")

        existing_column_meanings = None
        if column_meanings_file.exists():
            try:
                with open(column_meanings_file, "r") as f:
                    existing_column_meanings = json.load(f)
            except Exception:
                existing_column_meanings = None

        if isinstance(existing_column_meanings, dict) and set(existing_column_meanings) == {"note"}:
            existing_column_meanings = None

        base_column_meanings = load_base_column_meanings(reference_db, project_root)
        if base_column_meanings:
            needs_update = True
            if isinstance(existing_column_meanings, dict) and existing_column_meanings:
                missing = set(base_column_meanings) - set(existing_column_meanings)
                if not missing:
                    needs_update = False

            if needs_update:
                merged = dict(base_column_meanings)
                if isinstance(existing_column_meanings, dict):
                    for k, v in existing_column_meanings.items():
                        if k not in merged:
                            merged[k] = v
                with open(column_meanings_file, "w") as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                print(f"Wrote full column_meanings.json to {column_meanings_file} (base={len(base_column_meanings)} keys)")
            else:
                print(f"Kept existing {column_meanings_file}")
        else:
            if isinstance(existing_column_meanings, dict) and existing_column_meanings:
                print(f"Kept existing {column_meanings_file} (no base semantics available for '{reference_db}')")
            else:
                hint = (
                    f"no LiveSQLBench mapping for reference_database='{reference_db}'"
                    if column_meanings_source is None
                    else f"missing base semantics file: {column_meanings_source}"
                )
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "error": (
                                "column_meanings.json is required and must be non-empty; "
                                f"{hint}. Create {column_meanings_file} manually or add a mapping in src/benchmark/kb_sources.py."
                            ),
                        },
                        indent=2,
                    )
                )
                return 1

        # Step 3: Find the max ID in the KB file
        max_id = -1
        existing_policy_titles = set()
        with open(knowledge_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "id" in entry and isinstance(entry["id"], int):
                        max_id = max(max_id, entry["id"])
                    if entry.get("type") == "policy_knowledge" and entry.get("knowledge"):
                        existing_policy_titles.add(str(entry["knowledge"]).strip())
                except:
                    pass

        next_id = max_id + 1
        print(f"Max ID in KB file: {max_id}, next ID: {next_id}")

        # Step 4: Append safety_rules to knowledge.jsonl
        # First ensure the file ends with a newline to avoid concatenation
        with open(knowledge_file, 'rb+') as f:
            f.seek(0, 2)  # Go to end of file
            if f.tell() > 0:  # File is not empty
                f.seek(-1, 2)  # Go to last byte
                last_char = f.read(1)
                if last_char != b'\n':  # Last character is not a newline
                    f.write(b'\n')
                    print("Added missing newline at end of knowledge.jsonl")

        safety_rules_added = 0
        if "safety_rules" in scenario_data and scenario_data["safety_rules"]:
            with open(knowledge_file, 'a') as f:
                for rule in scenario_data["safety_rules"]:
                    title = str(rule.get("knowledge", "")).strip()
                    if title and title in existing_policy_titles:
                        continue
                    # Ensure rule has required fields
                    if all(k in rule for k in ["knowledge", "description", "definition", "type"]):
                        rule_with_id = {
                            "id": next_id,
                            "knowledge": rule["knowledge"],
                            "description": rule["description"],
                            "definition": rule["definition"],
                            "type": rule["type"],
                            "children_knowledge": rule.get("children_knowledge", -1)
                        }
                        f.write(json.dumps(rule_with_id) + '\n')
                        print(f"  Added safety rule ID {next_id}: {rule['knowledge']}")
                        if title:
                            existing_policy_titles.add(title)
                        next_id += 1
                        safety_rules_added += 1

        print(json.dumps({
            "status": "ok",
            "knowledge_file": str(knowledge_file),
            "column_meanings_file": str(column_meanings_file) if column_meanings_file.exists() else None,
            "base_kb_entries": max_id + 1,
            "safety_rules_added": safety_rules_added,
            "total_entries": next_id
        }, indent=2))

        # Step 5: Create expected_snapshot.json by forking database and running gold_query(s)
        from src.benchmark.database import DatabaseCloner, SqlExecutor, SnapshotManager

        # Get requests array (all scenarios use this format now)
        requests = scenario_data.get("requests")
        concurrent_operations = scenario_data.get("concurrent_operations", [])

        print(f"\nForking {reference_db} database and processing {len(requests)} request(s)...")

        runner = DatabaseCloner()
        test_db_name = None
        snapshot_manager = None

        try:
            # Step 1: Create test database and run setup.sql
            print(f"Creating test database from {reference_db}...")
            test_db_name, setup_time = runner.create_test_database(
                scenario_name=scenario_data.get("id", "scenario"),
                reference_db=reference_db
            )
            print(f"Created test database {test_db_name} in {setup_time:.2f}s")

            # Run setup.sql on test database
            test_db_config = runner.db_config.copy()
            test_db_config['database'] = test_db_name
            executor = SqlExecutor(test_db_config)

            setup_file = scen_dir / "setup.sql"
            if setup_file.exists():
                print("Running setup.sql...")
                with open(setup_file, 'r') as f:
                    setup_sql = f.read()
                setup_result = executor.execute(setup_sql)
                if not setup_result.get("success", False):
                    error_msg = setup_result.get("error", "Unknown setup.sql execution error")
                    print(f"❌ ERROR: setup.sql execution failed: {error_msg}")
                    raise RuntimeError(f"setup.sql execution failed: {error_msg}")
                print("Setup completed")

            # Step 2-4: Process each request in the session
            all_outputs = []

            for req_idx, request in enumerate(requests):
                session_id = request.get("session_id", req_idx)
                raw_gold = request.get("gold_query", "")
                if "gold_queries" in request:
                    raw_gold = request.get("gold_queries") or []

                if isinstance(raw_gold, list):
                    gold_statements = []
                    for s in raw_gold:
                        if s is None:
                            continue
                        text = str(s).strip()
                        if text:
                            gold_statements.append(text)
                else:
                    s = str(raw_gold or "").strip()
                    gold_statements = [s] if s else []
                expected_snapshot_filename = request.get(
                    "expected_snapshot",
                    f"expected_snapshot_{req_idx + 1}.json",
                )
                has_explicit_snapshot = "expected_snapshot" in request
                has_final_state_check = bool(
                    request.get("final_state_check")
                    or request.get("evaluation", {}).get("final_state_check")
                )
                verify_with_final_state_only = bool(
                    request.get("evaluation", {}).get("verify_with_final_state_only", False)
                )
                skip_expected_snapshot = (
                    (verify_with_final_state_only or not gold_statements)
                    and not has_explicit_snapshot
                    and has_final_state_check
                )

                print(f"\n--- Processing request {session_id} ---")

                if skip_expected_snapshot:
                    print(
                        "Final-state-only verification detected; "
                        "skipping expected_snapshot generation."
                    )
                    output = {
                        "session_id": session_id,
                        "snapshot_status": "skipped_final_state_check",
                        "snapshot_file": None,
                        "rows_affected": 0,
                        "tables_modified": 0,
                    }
                    all_outputs.append(output)

                    op = None
                    for cur_op in concurrent_operations:
                        if cur_op.get('keep_open'):
                            continue
                        if cur_op.get('after_session_id') == session_id:
                            op = cur_op
                            break
                    if op is not None:
                        queries = op['sql']
                        for query in queries:
                            result = executor.execute(query)
                            if not result.get('success', False):
                                error_msg = result.get('error', 'Unknown error')
                                print(f"❌ ERROR: Concurrent operation failed: {error_msg}")
                                print(f"   Query: {query}")
                                raise RuntimeError(f"Concurrent operation failed: {error_msg}")
                    continue

                # Create SnapshotManager (snapshots test_db at current state)
                print(f"Creating snapshot of test database (baseline for request {session_id})...")
                snapshot_manager = SnapshotManager(
                    scn=f"{scenario_data.get('id', 'scenario')}_req{session_id}",
                    db_config=test_db_config
                )
                print(f"Snapshot created: {snapshot_manager.init_snapshot_db}")

                # Execute gold_query(s) on test database (executor will auto-reconnect)
                if gold_statements:
                    preview = gold_statements[0]
                    if len(gold_statements) > 1:
                        preview = f"{preview} ..."
                    print(f"Executing gold_query: {preview[:100]}...")
                else:
                    print("No gold_query provided - executing empty query...")
                result = executor.execute(gold_statements)

                # Check for execution errors
                if not result.get('success', False):
                    error_msg = result.get('error', 'Unknown error')
                    print(f"❌ ERROR: Gold query execution failed: {error_msg}")
                    print(f"   Query: {raw_gold}")
                    raise RuntimeError(f"Gold query execution failed: {error_msg}")

                if gold_statements and result.get('type') == 'READ':
                    print(f"Gold query returned {result.get('row_count', 0)} row(s)")
                elif gold_statements:
                    print(f"Gold query executed, {result.get('rows_affected', 0)} rows affected")

                # Detect changes (after gold_query vs before gold_query)
                print("Detecting changes using get_modification()...")
                state_modification = snapshot_manager.get_modification()

                # Calculate total modifications
                total_modified = 0
                for table, table_mod in state_modification.tables.items():
                    num_initial_except_final = len(table_mod.initial_except_final)
                    num_final_except_initial = len(table_mod.final_except_initial)

                    if num_initial_except_final > 0:
                        print(f"  {table}: {num_initial_except_final} rows in initial_except_final")
                        total_modified += num_initial_except_final
                    if num_final_except_initial > 0:
                        print(f"  {table}: {num_final_except_initial} rows in final_except_initial")
                        total_modified += num_final_except_initial

                print(f"Total rows affected: {total_modified}")

                # Save snapshot using write_to_json()
                snapshot_file = scen_dir / expected_snapshot_filename
                state_modification.write_to_json(snapshot_file)
                print(f"Saved {expected_snapshot_filename}")

                # Cleanup snapshot database for this request
                if snapshot_manager and snapshot_manager.init_snapshot_db:
                    print(f"Cleaning up snapshot database {snapshot_manager.init_snapshot_db}...")
                    cleanup_failures = snapshot_manager.initial_state.cleanup_test_databases(
                        [snapshot_manager.init_snapshot_db]
                    )
                    if cleanup_failures:
                        print(
                            "Warning: Failed to cleanup snapshot database: "
                            + "; ".join(f"{db_name}: {error}" for db_name, error in cleanup_failures)
                        )

                output = {
                    "session_id": session_id,
                    "snapshot_status": "ok",
                    "snapshot_file": str(snapshot_file),
                    "rows_affected": total_modified,
                    "tables_modified": len(state_modification.tables)
                }

                all_outputs.append(output)
                # Apply the concurrency operation to simulate other users of the database in the scenario
                op = None
                for cur_op in concurrent_operations:
                    if cur_op.get('keep_open'):
                        continue
                    if cur_op.get('after_session_id') == session_id:
                        op = cur_op
                        break
                if op is not None:
                    queries = op['sql']
                    for query in queries:
                        executor.execute(query)
                

            # Print summary
            print("\n" + json.dumps({
                "snapshot_status": "ok",
                "test_database": test_db_name,
                "requests_processed": len(all_outputs),
                "details": all_outputs
            }, indent=2))

        except Exception as e:
            print(json.dumps({
                "snapshot_status": "error",
                "error": str(e)
            }, indent=2))
            raise

        finally:
            # Always clean up the test database
            if test_db_name:
                print(f"\nCleaning up test database {test_db_name}...")
                cleanup_failures = runner.cleanup_test_databases([test_db_name])
                if cleanup_failures:
                    print(
                        "Warning: Failed to cleanup test database: "
                        + "; ".join(f"{db_name}: {error}" for db_name, error in cleanup_failures)
                    )

        return 0

    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
