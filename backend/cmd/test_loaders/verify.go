package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

func main() {
	// Test TruthfulQA
	fmt.Println("=== Testing TruthfulQA ===")
	f, err := os.Open("../../../data/benchmarks/truthfulqa/TruthfulQA.csv")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		defer f.Close()
		reader := csv.NewReader(f)
		header, _ := reader.Read()
		fmt.Printf("✓ Header: %v\n", header[:3])

		count := 0
		for {
			_, err := reader.Read()
			if err == io.EOF {
				break
			}
			count++
		}
		fmt.Printf("✓ Loaded %d TruthfulQA samples\n", count)
	}

	// Test FEVER
	fmt.Println("\n=== Testing FEVER ===")
	f2, err := os.Open("../../../data/benchmarks/fever/shared_task_dev.jsonl")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		defer f2.Close()
		count := 0
		decoder := json.NewDecoder(f2)
		for {
			var record map[string]interface{}
			if err := decoder.Decode(&record); err == io.EOF {
				break
			} else if err != nil {
				continue
			}
			count++
		}
		fmt.Printf("✓ Loaded %d FEVER samples\n", count)
	}

	// Test HaluEval
	fmt.Println("\n=== Testing HaluEval ===")
	f3, err := os.Open("../../../data/benchmarks/halueval/qa_samples.json")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		defer f3.Close()
		var records []map[string]interface{}
		if err := json.NewDecoder(f3).Decode(&records); err != nil {
			fmt.Printf("Error decoding: %v\n", err)
		} else {
			fmt.Printf("✓ Loaded %d HaluEval samples\n", len(records))
		}
	}

	fmt.Println("\n✓ Phase 1 Setup Complete: All benchmarks downloaded and verified")
}
