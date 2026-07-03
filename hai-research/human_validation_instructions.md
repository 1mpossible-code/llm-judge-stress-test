# Human perturbation validation instructions

This is a short human-review task for checking whether perturbations preserve the meaning of the original text.

## What you will rate

For each item, you will see:

1. the original text
2. a perturbed version of the same text
3. three 1--5 ratings

Ratings:

- **Meaning preservation**: 1 = changes meaning heavily, 5 = preserves meaning very well
- **Framing-only score**: 1 = adds substantial new content, 5 = adds only context/framing
- **Direct offensiveness-change score**: 1 = does not directly change offensiveness, 5 = directly changes offensiveness

High scores on the first two and low scores on the third are best.

## Recommended setup

Best option for the paper: ask **2 reviewers to each complete the full set**. From the `hai-research` directory, each reviewer runs:

```bash
uv run python scripts/human_perturbation_validation.py annotate --annotator-id YOUR_NAME_OR_ID
```

This gives two independent ratings for all 270 original/perturbed pairs. The script randomizes order, hides item/perturbation metadata by default, and repeats about 10% of pairs later as blinded duplicate reliability checks. So each full reviewer will see about 297 screens, not exactly 270.

If the full set is too much, use shards as a lighter fallback. If three people are helping, use one shard per person:

```bash
# Person 1
uv run python scripts/human_perturbation_validation.py annotate --annotator-id rater01 --num-shards 3 --shard-index 0

# Person 2
uv run python scripts/human_perturbation_validation.py annotate --annotator-id rater02 --num-shards 3 --shard-index 1

# Person 3
uv run python scripts/human_perturbation_validation.py annotate --annotator-id rater03 --num-shards 3 --shard-index 2
```

Each shard is about 90 primary ratings plus about 9 duplicate reliability checks. To do a quick test first:

```bash
uv run python scripts/human_perturbation_validation.py annotate --annotator-id test --max-tasks 5
```

You can quit anytime with `q`. Progress is saved automatically, and rerunning the same command resumes where you left off.

## Output

Each reviewer produces:

```text
human_validation/<annotator-id>.jsonl
```

Send that `.jsonl` file back to the project owner.

## Summarizing results

After collecting files from reviewers, place them in `human_validation/` and run:

```bash
uv run python scripts/human_perturbation_validation.py summarize --input-glob 'human_validation/*.jsonl'
```

This writes summary files to:

```text
paper_artifacts/human_validation/
```

The main semantic-validation summary excludes blinded duplicates. Duplicate ratings are used separately to estimate intra-rater consistency.
