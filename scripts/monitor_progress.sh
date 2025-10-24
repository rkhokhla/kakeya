#!/bin/bash
# Monitor LLM generation progress

echo "=== LLM Generation Progress Monitor ==="
echo ""

# Check TruthfulQA
if [ -f data/checkpoints/truthfulqa_checkpoint.json ]; then
    echo "📊 TruthfulQA:"
    completed=$(jq -r '.stats.total_samples' data/checkpoints/truthfulqa_checkpoint.json 2>/dev/null || echo "0")
    cost=$(jq -r '.stats.total_cost' data/checkpoints/truthfulqa_checkpoint.json 2>/dev/null || echo "0.0")
    echo "  Completed: $completed / 789 samples"
    echo "  Cost: \$$cost"
    echo "  Progress: $(awk "BEGIN {printf \"%.1f%%\", ($completed/789)*100}")"
fi

echo ""

# Check FEVER
if [ -f data/checkpoints/fever_checkpoint.json ]; then
    echo "📊 FEVER:"
    completed=$(jq -r '.stats.total_samples' data/checkpoints/fever_checkpoint.json 2>/dev/null || echo "0")
    cost=$(jq -r '.stats.total_cost' data/checkpoints/fever_checkpoint.json 2>/dev/null || echo "0.0")
    echo "  Completed: $completed / 2,500 samples"
    echo "  Cost: \$$cost"
    echo "  Progress: $(awk "BEGIN {printf \"%.1f%%\", ($completed/2500)*100}")"
fi

echo ""

# Check HaluEval
if [ -f data/checkpoints/halueval_checkpoint.json ]; then
    echo "📊 HaluEval:"
    completed=$(jq -r '.stats.total_samples' data/checkpoints/halueval_checkpoint.json 2>/dev/null || echo "0")
    cost=$(jq -r '.stats.total_cost' data/checkpoints/halueval_checkpoint.json 2>/dev/null || echo "0.0")
    echo "  Completed: $completed / 5,000 samples"
    echo "  Cost: \$$cost"
    echo "  Progress: $(awk "BEGIN {printf \"%.1f%%\", ($completed/5000)*100}")"
fi

echo ""
echo "=== Total Summary ==="

total_completed=0
total_cost=0.0

for checkpoint in data/checkpoints/*_checkpoint.json; do
    if [ -f "$checkpoint" ]; then
        samples=$(jq -r '.stats.total_samples' "$checkpoint" 2>/dev/null || echo "0")
        cost=$(jq -r '.stats.total_cost' "$checkpoint" 2>/dev/null || echo "0.0")
        total_completed=$((total_completed + samples))
        total_cost=$(awk "BEGIN {printf \"%.4f\", $total_cost + $cost}")
    fi
done

echo "Total Completed: $total_completed / 8,289 samples"
echo "Total Cost: \$$total_cost"
echo "Overall Progress: $(awk "BEGIN {printf \"%.1f%%\", ($total_completed/8289)*100}")"

echo ""
echo "💡 Latest logs:"
echo "   tail -f logs/truthfulqa_generation.log"
echo "   tail -f logs/fever_generation.log"
echo "   tail -f logs/halueval_generation.log"
