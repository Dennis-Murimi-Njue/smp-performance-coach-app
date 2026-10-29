# SMP Performance Coach

APyton application that trains on historical SMP fitness data, predicts whether a new day will hit the 10,000-steps goal and generates a coaching message based on the prediction and input values

It combines a trained machine learning model with a rule-based coaching layer, and includes a path to upgrade the coaching to live GPT-4o responses.

---

## What it does

1. **Trains** a `RandomForestClassifier` on 28 days of logged data.
2. **Predicts** for each new day: goal hit or missed, with a confidence score.
3. **Coaches** using a lookup table of 8 pre-written messages, chosen by three yes/no factors: goal hit, sleep OK (7h or more), water OK (8 glasses or more).
4. **Reports** each day individually, then prints a summary comparing hit days against miss days.

## Inputs and label

| Feature | Meaning |
|---|---|
| `sleep_hr` | Hours slept |
| `water_glasses` | Glasses of water drunk |
| `bench_kg` | Bench press weight in kg |

**Label:** `1` = hit 10,000 steps, `0` = did not.

## Requirements

- Python 3.9+
- `numpy`
- `scikit-learn`

```bash
pip install numpy scikit-learn
```

Note the install name is `scikit-learn`, but you import it as `sklearn`.

## Run it

Save the code as `smp_coach.py`, then:

```bash
python smp_coach.py
```

Run it as a file rather than pasting it line by line into the Python shell, so that earlier lines (imports, trained model) are always defined.

## How it works

```
 28-day log ──► train_test_split (80/20) ──► RandomForestClassifier (100 trees)
                                                      │
 new day (sleep, water, bench) ──► analyze_day() ─────┤
                                                      ▼
                                      prediction + confidence
                                                      │
                                                      ▼
                          get_coaching_message()  (8-template lookup)
                                                      │
                                                      ▼
                                   result dictionary ──► report + summary
```

### Key functions

**`analyze_day(sleep_hr, water_glasses, bench_kg, day_label=None)`**
Runs the full pipeline for one day. Returns a dictionary with `label`, the three inputs, `hit_goal` (bool), `confidence` (0 to 1), and `coaching` (text). If no label is given, it defaults to `"Day"`.

**`get_coaching_message(sleep, water, hit_goal)`**
Builds the key `(hit_goal, sleep >= 7.0, water >= 8)` and looks it up in `COACHING_TEMPLATES`. Three yes/no factors give 2 x 2 x 2 = 8 combinations, and all 8 are covered, so there is no uncovered case.

### Batch processing

Days to analyze are listed as `(label, sleep, water, bench)` tuples in `incoming_days`. A list comprehension runs `analyze_day()` on all of them, then the summary section counts hits, averages confidence, and compares average sleep and water on hit days versus miss days.

## Sample output

Real output from the current code (Days 29 to 37), abbreviated:

```
==============================================================
    SMP PERFORMANCE COACH  |  DAILY REPORTS
==============================================================

Day 29
  Outcome:  HIT GOAL (100% confidence)
  Inputs:   sleep=8.0h  water=10gl  bench=92kg
  Coach:    Strong inputs, strong output. Baseline is locked in. Keep this pattern consistent.

Day 31
  Outcome:  HIT GOAL (60% confidence)
  Inputs:   sleep=7.2h  water=7gl  bench=84kg
  Coach:    Hit the goal despite low water. Sleep is the primary driver. Close the hydration gap tomorrow.

...

==============================================================
    SUMMARY
==============================================================

  Days analyzed:    9
  Goals hit:        5 / 9  (56%)
  Goals missed:     4 / 9  (44%)
  Avg confidence:   96%

  On HIT days:
    avg sleep:  8.1h  |  avg water: 9.8 glasses

  On MISS days:
    avg sleep:  5.5h  |  avg water: 4.2 glasses

  HIT days averaged +2.6h more sleep and +5.6 more glasses of water
```

Day 31 (7.2h sleep, 7 glasses water) is the borderline case: the model predicts a hit at only 60% confidence.

## Known limitations

- **Tiny dataset.** 28 rows, split into 22 for training and 6 for testing. One test prediction changes accuracy by about 17 points, so any accuracy figure is noisy.
- **Clean, hand-built data.** The pattern is very consistent, so the model looks more confident than it would on real, messy logs.
- **Out-of-range inputs.** The model has never seen values outside its training range (sleep 5.5 to 8.5h, water 4 to 11 glasses). Predictions for something like 12 glasses of water are extrapolation and less trustworthy.
- **Coaching ignores bench press.** The template lookup uses only sleep, water, and the predicted outcome. Bench is used by the model but not by the coaching text.
- **Coaching reacts to the prediction, not the real result.** It speaks as if the predicted outcome already happened.

## Upgrading to the production version

The pipeline is designed so each piece can be swapped without rewriting the rest:

| Piece | This version | Production version |
|---|---|---|
| Data | Hardcoded `np.array` | `pd.read_csv("smp_log.csv")` with `pandas` |
| Model | Retrained every run | Saved once with `joblib.dump`, reloaded with `joblib.load` |
| Coaching | 8 fixed templates | Live GPT-4o call via the OpenAI API |

Extra setup for the production version:

```bash
pip install pandas joblib openai python-dotenv
```

1. Create a `.env` file in the project folder containing `OPENAI_API_KEY=your-key-here` (no quotes, no spaces around `=`).
2. Add `.env` to `.gitignore` so the key is never committed.
3. Provide `smp_log.csv` with columns `sleep_hr`, `water_glasses`, `bench_kg`, and `steps`.
4. Wrap the model's JSON or text handling in `try/except` where parsing is involved, since live model output is not guaranteed to be perfectly formatted.

The OpenAI API is a paid service, billed per call, so call the coaching function sparingly (for example once per day rather than on every prediction).

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` in the same Python environment VS Code is using (check the interpreter in the bottom-right corner). |
| `NameError` after an import error | A cascade from the failed import. Fix the import and run the whole file again. |
| `OpenAIError: Missing credentials` | `.env` was not found or the key name is wrong. Confirm the file is named exactly `.env`, sits in the folder you run from, and contains `OPENAI_API_KEY=...`. |

## Ideas for next steps

- Replace the hardcoded data with a real daily log in CSV.
- Add `classification_report` and a confusion matrix to evaluate more honestly than a single accuracy number.
- Use cross-validation instead of one 80/20 split.
- Pass `bench_kg` and `confidence` into the coaching logic.
- Guard against empty lists and bad inputs.
