package main

import (
	"fmt"
	"log"

	"github.com/fractal-lba/kakeya/internal/eval"
)

func main() {
	loader := eval.NewBenchmarkLoader("../../data/benchmarks")

	// Test TruthfulQA
	fmt.Println("=== Testing TruthfulQA Loader ===")
	tqa, err := loader.LoadTruthfulQA()
	if err != nil {
		log.Printf("TruthfulQA error: %v", err)
	} else {
		fmt.Printf("✓ Loaded %d TruthfulQA samples\n", len(tqa))
		if len(tqa) > 0 {
			fmt.Printf("  Sample: ID=%s, Prompt=%s\n", tqa[0].ID, tqa[0].Prompt[:80]+"...")
		}
	}

	// Test FEVER
	fmt.Println("\n=== Testing FEVER Loader ===")
	fever, err := loader.LoadFEVER(1000) // Load 1000 samples
	if err != nil {
		log.Printf("FEVER error: %v", err)
	} else {
		fmt.Printf("✓ Loaded %d FEVER samples\n", len(fever))
		if len(fever) > 0 {
			fmt.Printf("  Sample: ID=%s, Response=%s\n", fever[0].ID, fever[0].Response[:80]+"...")
		}
	}

	// Test HaluEval
	fmt.Println("\n=== Testing HaluEval Loader ===")
	halueval, err := loader.LoadHaluEval()
	if err != nil {
		log.Printf("HaluEval error: %v", err)
	} else {
		fmt.Printf("✓ Loaded %d HaluEval samples\n", len(halueval))
		if len(halueval) > 0 {
			fmt.Printf("  Sample: ID=%s, Prompt=%s\n", halueval[0].ID, halueval[0].Prompt[:80]+"...")
		}
	}

	// Summary
	fmt.Println("\n=== Summary ===")
	total := len(tqa) + len(fever) + len(halueval)
	fmt.Printf("Total samples available: %d\n", total)
	fmt.Printf("  - TruthfulQA: %d\n", len(tqa))
	fmt.Printf("  - FEVER: %d\n", len(fever))
	fmt.Printf("  - HaluEval: %d\n", len(halueval))
}
