# Be CARE-ful with Text-to-SQL Benchmarks

Official code release for **Be CARE-ful with Text-to-SQL Benchmarks** (NeurIPS 2026 Evaluations and Datasets Track, poster).

CARE tests whether LLM agents can do database work safely, not just produce correct SQL. Agents work on real PostgreSQL databases through tools. Each task checks one of four dimensions:

| Dimension | Task folder |
|---|---|
| **C**oncurrency Control | `concurrency/` |
| **A**ccess Control & Safety Policies | `access_control_and_security/` |
| **R**ules & Integrity | `business_and_integrity_rules/` |
| **E**fficiency | `efficiency/` |

This repository contains the code for the benchmark.

## Repository layout

```
eval.py                      # Runs an agent on tasks and grades the result
src/
  benchmark/                 # Agent loop, tools, database cloning and snapshots,
                             # task loading, result comparison, judge and simulated user
  agents/                    # LLM agent
  llm/                       # Model clients (OpenAI, Anthropic, Amazon Bedrock);
                             # model presets live in src/llm/factory.py
  prepare_scenario.py        # Fills a task folder's knowledge/column-meaning files and
                             # derives expected state from the gold queries
  config.py                  # Database connection settings
scripts/setup/               # Create template databases from SQL dumps; clean up
                             # leftover test databases
tests/                       # Unit tests (no database needed)
```

## Requirements

- Python 3 (developed with Python 3.13)
- A PostgreSQL server with the benchmark's reference databases loaded
- API access to the models you want to evaluate (OpenAI, Anthropic, or Amazon Bedrock)

## Setup

```bash
git clone https://github.com/haonanw98/careful-bench.git
cd careful-bench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

- **Database.** `DB_MODE=local` uses the `POSTGRES_*` settings; `DB_MODE=custom` uses the `DB_*` settings.
- **Models.** Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. For Bedrock, set `AWS_REGION` and either `AWS_PROFILE` or AWS access keys.

## Data

Place the released data in the repository root. These paths are listed in `.gitignore`.

- `scenarios/<dimension>/<family>/<task_id>/` holds one folder per task: `scenario.json`, `knowledge.jsonl`, `column_meanings.json`, and an optional `setup.sql`.
- `livesqlbench-base-full-v1/` holds the per-database knowledge and column meanings that the task loader and `prepare_scenario.py` read. To keep it elsewhere, set `LIVESQLBENCH_ROOT`.
- **Reference databases.** Each task names a `reference_database`, which must exist on your PostgreSQL server. To create one from a SQL dump:

  ```bash
  python scripts/setup/setup_external_postgres_db.py --mode local --db-name <name> --sql-file <dump.sql>
  ```

## Running

Run from the repository root:

```bash
# One task, all of its requests
python eval.py --scenario concurrency/stale_confirmation_scope_drift/100_virtual_idol_same_count_support_drift_native --all-requests

# Every task whose path contains "efficiency", with a chosen model
python eval.py --scenario efficiency --scenario-fuzzy --model <model>

# Every task under scenarios/
python eval.py
```

Common options (`python eval.py --help` lists all of them):

| Option | Meaning |
|---|---|
| `--model`, `-m` | Model under test. Presets are in `src/llm/factory.py`. |
| `--aux-model` | Model for the judge and the simulated user (default `gpt-5.2`). |
| `--scenario`, `-s` | Exact task path under `scenarios/`. Add `--scenario-fuzzy` to match part of a path or a task name. |
| `--request-index` / `--all-requests` | Run one request of a multi-request task, or all of them. |
| `--total-budget`, `--max-steps` | Override the per-request tool budget and step limit. |

Cloned test databases are named `test_*`. To find ones left behind by interrupted runs, use `scripts/setup/cleanup_stale_test_databases.py`.

## How a task is graded

For each task, `eval.py` runs these steps:

1. **Clone the database.** The task's reference database is copied into a fresh test database (`CREATE DATABASE ... TEMPLATE`), so runs can go in parallel.
2. **Prepare.** The task's `setup.sql` is applied, if there is one.
3. **Run the agent.** The agent works on each request with these tools:
   - `explore_database` and `test_query`, which run in read-only transactions;
   - `read_knowledge` (business rules and safety policies);
   - `read_semantics` (column meanings);
   - `ask_user` (a simulated user that answers only from a fixed set of answers written for the task);
   - `write_db`, which commits changes.

   The agent ends each request with a final decision: execute or reject. The runner caps steps and tool use.
4. **Check correctness.**
   - The agent's decision (execute or reject) must match the expected one.
   - For write tasks, the database change is compared with the change produced by the gold queries.
   - For read tasks, the agent's result is compared with the gold result.
   - Efficiency tasks also compare the agent's query time with the gold query's (by default, it must be within 5×), and can check the query plan.
5. **Check understanding.** If the task has follow-up questions, a judge model scores the agent's answers using only the tool evidence and the task's knowledge.

A task passes only if both checks pass.

## Outputs

Each run writes to `logs/sessions/<session>/`:

- `main.log`, the run log;
- `eval_result_*.json`, the per-task results;
- `metadata.json`, the session details.

## Tests

```bash
python -m pytest tests
```

## License

- **Code:** MIT License; see [`LICENSE`](LICENSE).
- **Task data:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), because the tasks build on LiveSQLBench and BIRD-Interact data, which are CC BY-SA 4.0.
- **Upstream data:** CAREFUL builds on databases from Spider 2.0 and BIRD-Interact (LiveSQLBench), which keep their own licenses: [Spider 2.0](https://github.com/xlang-ai/Spider2) (MIT), [LiveSQLBench](https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1) (CC BY-SA 4.0; the dataset card's metadata says CC BY 4.0, but its license section and the LiveSQLBench GitHub README say CC BY-SA 4.0), and [BIRD-Interact](https://huggingface.co/datasets/birdsql/bird-interact-full) (CC BY-SA 4.0). Follow their terms when you download and use those databases.
