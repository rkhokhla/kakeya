# Setting Up OpenAI API Key for Phase 3

## Quick Start

1. **Get your OpenAI API key:**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (starts with `sk-proj-...` or `sk-...`)

2. **Set the environment variable:**

   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

3. **Verify it's set:**

   ```bash
   source venv/bin/activate
   python3 -c "import os; print('API key:', os.getenv('OPENAI_API_KEY', 'NOT SET')[:10] + '...')"
   ```

4. **Run test generation (2 samples, costs ~$0.003):**

   ```bash
   source venv/bin/activate
   python3 scripts/generate_llm_outputs.py --benchmark truthfulqa --limit 2
   ```

## Alternative: Use .env file

1. **Create .env file:**

   ```bash
   cp .env.example .env
   nano .env  # Edit with your API key
   ```

2. **Load environment:**

   ```bash
   source .env
   ```

## Cost Estimates

**Test run (2 samples):** ~$0.003
**TruthfulQA (789 samples):** ~$1.18
**FEVER (2,500 samples):** ~$3.75
**HaluEval (5,000 samples):** ~$7.50
**Total for all benchmarks:** ~$12.43

Actual costs may vary based on:
- Token count per response (estimated: 150 tokens)
- Model pricing (GPT-4 Turbo: $0.01/1K input tokens)
- Number of retries on failures

## Safety Features

The generation script includes:
- Cost tracking (printed after each sample)
- Checkpoint/resume (can pause and resume anytime)
- Progress bars with ETA
- Rate limiting (0.5s between requests)
- Dry-run mode for testing without API calls

## Troubleshooting

**"OPENAI_API_KEY not set" error:**
```bash
# Check if it's set
echo $OPENAI_API_KEY

# If empty, set it again
export OPENAI_API_KEY='your-key-here'
```

**Rate limit errors:**
- Script automatically retries with exponential backoff
- Wait times: 2s, 4s, 8s before giving up
- Check your OpenAI account tier and limits

**Out of credits:**
- Check your OpenAI account balance
- Add credits at: https://platform.openai.com/account/billing

## Next Steps After Key Setup

Run the full pipeline:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate outputs for all benchmarks (~8-15 hours, ~$12-15)
python3 scripts/generate_llm_outputs.py --benchmark all

# Or run one at a time:
python3 scripts/generate_llm_outputs.py --benchmark truthfulqa   # 789 samples
python3 scripts/generate_llm_outputs.py --benchmark fever        # 2,500 samples
python3 scripts/generate_llm_outputs.py --benchmark halueval     # 5,000 samples
```

The script will:
- Save outputs to `data/llm_outputs/<benchmark>_outputs.jsonl`
- Save checkpoints to `data/checkpoints/<benchmark>_checkpoint.json`
- Print progress and cost after each sample
- Allow resuming if interrupted (just run the same command again)
