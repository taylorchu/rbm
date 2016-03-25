package rbm

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func permFloat64(input [][]float64, perm []int) [][]float64 {
	output := make([][]float64, len(input))
	for i := 0; i < len(perm); i++ {
		output[i] = input[perm[i]]
	}
	return output
}

func permInt(input []int, perm []int) []int {
	output := make([]int, len(input))
	for i := 0; i < len(perm); i++ {
		output[i] = input[perm[i]]
	}
	return output
}

func normalizer(input [][]float64) func(...[]float64) {
	if len(input) == 0 {
		return func(_ ...[]float64) {
			return
		}
	}

	mean := make([]float64, len(input[0]))
	stdev := make([]float64, len(input[0]))
	nonBinary := make([]bool, len(input[0]))

	// all non-binary rows are zero
	allZero := make([]bool, len(input))
	for i := 0; i < len(input[0]); i++ {
		for j := 0; j < len(input); j++ {
			switch input[j][i] {
			case 0:
			case 1:
			default:
				nonBinary[i] = true
			}
		}
	}

	for i := 0; i < len(input); i++ {
		allZero[i] = true

		for j := 0; j < len(input[0]); j++ {
			if nonBinary[j] && input[i][j] != 0 {
				allZero[i] = false
			}
		}
	}

	for i := 0; i < len(input[0]); i++ {
		if !nonBinary[i] {
			fmt.Printf("col %d: binary\n", i+1)
			continue
		}
		var count float64
		for j := 0; j < len(input); j++ {
			if allZero[j] {
				continue
			}
			mean[i] += input[j][i]
			count++
		}
		mean[i] /= count

		for j := 0; j < len(input); j++ {
			if allZero[j] {
				continue
			}
			stdev[i] += (input[j][i] - mean[i]) * (input[j][i] - mean[i])
		}
		stdev[i] /= count
		stdev[i] = math.Sqrt(stdev[i])

		fmt.Printf("col %d: mean %f stdev %f\n", i+1, mean[i], stdev[i])
	}

	return func(v ...[]float64) {
		for i := 0; i < len(v); i++ {
			for j := 0; j < len(v[i]); j++ {
				if nonBinary[j] {
					if allZero[i] {
						v[i][j] = -3
					} else {
						v[i][j] -= mean[j]
						v[i][j] /= stdev[j]
					}
				} else {
					switch v[i][j] {
					case 0:
						v[i][j] = -3
					case 1:
						v[i][j] = 3
					}
				}
			}
		}
	}
}

func TestStackedClassifierTrain(t *testing.T) {
	c, err := NewStackedClassifier(true, 4, 8, 2, 4)
	if err != nil {
		t.Fatal(err)
	}

	input := [][]float64{
		{4, 5, 6, 1, 0},
		{2, 3, 4, 1.5, 0},
		{1, 2, 3, 1.3, 0},
		{2, 3, 5, 4.3, 1},
		{2, 3, 5, 5.5, 1},
		{2, 3, 5, 5, 1},
		{2, 4, 5, 6, 1},
		{2, 3, 4, 1, 0},
		{1, 3, 4, 1, 0},
		{2, 4, 6, 6.3, 1},
		{1, 3, 4, 6.3, 1},
		{2, 3, 4, 5, 1},
		{2, 3, 5, 4.3, 1},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 1, 2, 1.5, 0},
		{0, 0, 0, 0, 0},
		{2, 2.5, 3, 1, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{1, 1.5, 2.5, 1.5, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 1, 2, 1, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 1, 2, 1.5, 1},
		{2, 2.5, 3, 1, 0},
		{0, 0, 0, 0, 0},
		{1, 1.5, 3, 1.5, 1},
		{0.5, 1, 2, 1, 0},
		{1, 2, 2.5, 1.2, 0},
		{1.2, 1.7, 2.1, 2, 0},
		{1, 4, 5, 2, 1},
		{1.2, 2.5, 3, 2, 1},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{1, 2, 3, 1.1, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 1, 1.5, 1.5, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{1.5, 2, 2.5, 1, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{1.5, 2, 2.5, 1, 0},
		{0.8, 1, 2.2, 0.8, 0},
		{0.8, 2, 2.2, 0.8, 0},
		{0, 0.8, 1.2, 1.2, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
	}

	perm := rand.Perm(len(input))
	input = permFloat64(input, perm)

	n := normalizer(input)
	n(input...)
	output := []int{
		0,
		0,
		0,
		1,
		1,
		1,
		1,
		0,
		0,
		1,
		1,
		1,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		1,
		0,
		0,
		0,
		1,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
	}

	output = permInt(output, perm)

	c.Train(input, output, &Option{
		BatchSize: 10,
		Iteration: 2000,
		GibbsStep: 50,
	})

	for i := 0; i < len(input); i++ {
		got := c.Classify(input[i])
		want := output[i]
		if !reflect.DeepEqual(got, want) {
			t.Logf("%d: expect %v, got %v", perm[i]+1, want, got)
		}
	}
}

func TestStackedClassifierTrain2(t *testing.T) {
	c, err := NewStackedClassifier(true, 5, 10, 4, 8)
	if err != nil {
		t.Fatal(err)
	}

	input := [][]float64{
		{2, -2, 2, 0, 0},
		{2, -2, 3, 0, 0},
		{6, 0.5, 4.5, 0, 0},
		{4, 2, 7, 0, 0},
		{2, -1, 4, 0, 0},
		{3, 2, 5, 0, 0},
		{4, -2.5, 0, 0, 1},
		{4, -0.8, -0.3, 0, 1},
		{6, -2, 0, 0, 1},
		{3, -3, -1, 0, 1},
		{0, -6, -3, 0, 0},
		{4, -4, 0, 0, 1},
		{9, 2.5, 2.5, 1, 0},
		{12, 4, 4, 0, 1},
		{3, -4, 0.5, 1, 0},
		{3, -4, 0, 1, 0},
		{5, 1, 4.5, 0, 0},
		{5, 1, 5, 0, 0},
		{3, -3.5, 0, 0, 0},
		{1.5, -0.5, 3.5, 0, 0},
		{0, -4.5, -2, 0, 0},
		{6, -2, 0.5, 0, 0},
		{6, -2.5, 0.5, 0, 0},
		{5, 0.2, 2, 0, 0},
		{5, -1, 0.5, 1, 1},
		{6, -0.8, 0.4, 1, 0},
		{7, 0, 1.5, 1, 0},
		{11, 2, 2, 0, 1},
		{8, -0.2, 0.8, 1, 1},
		{5, -0.5, 0.5, 1, 0},
		{4, 0, 2, 0, 0},
		{3, 0, 2, 0, 0},
		{4, 0.3, 1.8, 0, 0},
		{3, 0, 2, 0, 0},
		{7, 1, 1.5, 0, 0},
		{8, 0.5, 1.3, 0, 0},
		{6, 0, 1, 0, 0},
		{5, -2, 2, 1, 1},
		{6, -1, 1, 0, 1},
		{6, 0, 2, 1, 1},
		{6, -2.5, 0.5, 0, 1},
		{0, -9, -3.5, 0, 0},
		{3, -4, 0, 0, 0},
		{3.5, -4, 0, 0, 0},
		{3, -3.5, 0, 0, 0},
		{4, -4.5, 0, 0, 0},
		{3, -3, 0, 0, 0},
		{7, -2, 0, 0, 1},
		{2, -2, 0.5, 0, 0},
		{3, 3, 7, 0, 0},
		{3, 1, 5, 0, 0},
		{6, -5.5, 3.5, 0, 1},
		{7, -4, 2, 0, 1},
		{3, -2.8, 0.7, 0, 0},
		{3, -5.5, 0.5, 0, 0},
		{3, -5, -0.5, 0, 0},
		{4, -4.5, -0.5, 0, 0},
		{4, -4, 0, 0, 0},
		{3, -3, 0.5, 0, 0},
		{9.5, -4.5, 0, 1, 0},
		{12, -5, -1, 1, 0},
		{7.5, -3.5, -0.5, 1, 0},
		{10, -2, 0, 0, 0},
		{10, 0, 3, 0, 0},
		{3, -5, -1, 0, 0},
		{2, -2.2, 0.5, 0, 0},
		{4, -2.8, 0.2, 0, 0},
		{2, -3.5, 0.5, 0, 0},
		{2, -5, -1, 0, 0},
		{2, -3, 0, 0, 0},
		{2, -0.5, 3, 0, 0},
		{3, 1, 3.5, 0, 0},
		{3, 0.5, 2.5, 0, 0},
		{3, 1, 3.8, 0, 0},
		{4, 1, 4.5, 0, 0},
		{4, -1, 4, 0, 0},
		{3, 0, 4, 0, 0},
		{6.5, -4.5, 1, 0, 1},
		{5, -4, 1, 1, 1},
		{4, -4, -0.5, 1, 1},
		{5, 1, 8, 0, 0},
	}

	perm := rand.Perm(len(input))
	input = permFloat64(input, perm)

	n := normalizer(input)
	n(input...)
	output := []int{
		1,
		1,
		2,
		2,
		2,
		2,
		3,
		3,
		3,
		3,
		0,
		3,
		3,
		3,
		3,
		3,
		2,
		2,
		1,
		2,
		0,
		1,
		1,
		2,
		3,
		3,
		3,
		3,
		3,
		3,
		2,
		2,
		2,
		2,
		2,
		2,
		2,
		3,
		3,
		3,
		3,
		0,
		1,
		1,
		1,
		1,
		1,
		3,
		1,
		2,
		2,
		3,
		3,
		1,
		1,
		1,
		1,
		1,
		1,
		3,
		3,
		3,
		1,
		2,
		1,
		1,
		1,
		1,
		1,
		1,
		2,
		2,
		2,
		2,
		2,
		2,
		2,
		3,
		3,
		3,
		2,
	}

	output = permInt(output, perm)

	c.Train(input, output, &Option{
		BatchSize: 10,
		Iteration: 2000,
		GibbsStep: 50,
	})

	for i := 0; i < len(input); i++ {
		got := c.Classify(input[i])
		want := output[i]
		if !reflect.DeepEqual(got, want) {
			t.Logf("%d: expect %v, got %v", perm[i]+1, want, got)
		}
	}
}
