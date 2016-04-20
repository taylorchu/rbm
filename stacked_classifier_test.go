package rbm

import (
	"bytes"
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

func expand(input [][]float64, add map[int]int) [][]float64 {
	output := make([][]float64, len(input))
	for i, v := range input {
		var v2 []float64
		for j, col := range v {
			if n, ok := add[j]; ok {
				one := make([]float64, n)
				one[int(col)] = 1
				v2 = append(v2, one...)
			} else {
				v2 = append(v2, col)
			}
		}
		output[i] = v2
	}
	return output
}

type colStats struct {
	mean  float64
	stdev float64
}

func normalize(input [][]float64, stats map[int]colStats, f func([]float64) bool) [][]float64 {
	output := make([][]float64, len(input))
	for i, v := range input {
		allZero := f != nil && f(v)
		v2 := make([]float64, len(v))
		for j, col := range v {
			if s, ok := stats[j]; ok {
				if allZero {
					v2[j] = -2
				} else {
					v2[j] = (col - s.mean) / s.stdev
				}
			} else {
				if col > 0 {
					v2[j] = 2
				} else {
					v2[j] = -2
				}
			}
		}
		output[i] = v2
	}
	return output
}

func normalizer(input [][]float64, f func([]float64) bool) [][]float64 {
	if len(input) == 0 {
		return nil
	}

	stats := make(map[int]colStats)
	for i := 0; i < len(input[0]); i++ {
		binary := true
		for j := 0; j < len(input); j++ {
			switch input[j][i] {
			case 0:
			case 1:
			default:
				binary = false
			}
		}
		if binary {
			fmt.Printf("col %d: binary\n", i+1)
			continue
		}
		var (
			mean  float64
			stdev float64
			count float64
		)
		for j := 0; j < len(input); j++ {
			if f != nil && f(input[j]) {
				continue
			}
			mean += input[j][i]
			count++
		}
		mean /= count

		for j := 0; j < len(input); j++ {
			if f != nil && f(input[j]) {
				continue
			}
			stdev += (input[j][i] - mean) * (input[j][i] - mean)
		}
		stdev /= count
		stdev = math.Sqrt(stdev)

		fmt.Printf("col %d: mean %f stdev %f\n", i+1, mean, stdev)

		stats[i] = colStats{
			mean:  mean,
			stdev: stdev,
		}
	}

	return normalize(input, stats, f)
}

func TestStackedClassifierMarshal(t *testing.T) {
	m, err := NewStackedClassifier(true, 4, 8, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	buf := new(bytes.Buffer)
	err = m.WriteTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := NewStackedClassifier(true, 4, 8, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	err = m2.ReadFrom(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m, m2) {
		t.Fatalf("not equal")
	}
}

func TestStackedClassifierTrain1(t *testing.T) {
	c, err := NewStackedClassifier(true, 4, 4, 2, 4)
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

	input = normalizer(input, func(v []float64) bool {
		for _, col := range v {
			if col != 0 {
				return false
			}
		}
		return true
	})
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
	c, err := NewStackedClassifier(true, 5, 5, 4, 4)
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
		{0, -3, -1.2, 0, 0},
		{0, -5, -3, 0, 0},
		{0, -5, -2, 0, 0},
		{0, -5, -3, 0, 0},
		{0, -5, -1.2, 0, 0},
		{0, -4, -1, 0, 0},
		{0, -3, -1.5, 0, 0},
		{0, -5, -1.8, 0, 0},
		{0, -4.5, -3, 0, 0},
		{0, -5, -3, 0, 0},
		{0, -5, -2, 0, 0},
		{0, -5, -1, 0, 0},
		{0, -5, -2, 0, 0},
		{0, -4, -1.5, 0, 0},
		{0, -4, -1.5, 0, 0},
		{0, -3, -1.2, 0, 0},
		{0, -4, -1, 0, 0},
		{0, -3.5, -1.2, 0, 0},
		{0, -4, -2, 0, 0},
		{0, -4, -2.2, 0, 0},
		{0, -5, -2, 0, 0},
		{0, -4, -2, 0, 0},
	}

	perm := rand.Perm(len(input))
	input = permFloat64(input, perm)

	input = normalizer(input, func(v []float64) bool {
		return v[0] == 0
	})
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

func TestStackedClassifierTrain3(t *testing.T) {
	c, err := NewStackedClassifier(true, 10, 10, 3, 20)
	if err != nil {
		t.Fatal(err)
	}

	input := [][]float64{
		{0, 1, 1, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{1, 1, 0, 0, 3, 1, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0, 3, 0, 0},
		{0, 1, 0, 1, 0, 5, 0, 0},
		{0, 1, 1, 0, 0, 2, 0, 0},
		{0, 0, 0, 1, 0, 5, 0, 0},
		{0, 1, 1, 0, 0, 5, 0, 0},
		{0, 1, 0, 1, 2, 1, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 5, 2, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 0, 7, 3, 0},
		{0, 1, 1, 0, 1, 2, 1, 0},
		{0, 0, 1, 0, 0, 4, 1, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 8, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 4, 0, 0},
		{0, 1, 0, 0, 0, 2, 1, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 2, 0, 0},
		{0, 1, 0, 0, 1, 8, 2, 0},
		{0, 1, 0, 1, 0, 0, 0, 0},
		{1, 1, 0, 0, 0, 4, 4, 0},
		{0, 0, 0, 0, 5, 0, 0, 0},
		{0, 1, 0, 0, 6, 0, 0, 0},
		{2, 1, 0, 0, 0, 2, 2, 0},
		{1, 0, 0, 0, 0, 6, 6, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 1, 1, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 1, 2, 0, 0, 0},
		{1, 0, 0, 0, 0, 3, 3, 0},
		{0, 1, 0, 1, 0, 2, 2, 0},
		{1, 0, 0, 0, 0, 3, 3, 0},
		{0, 0, 0, 1, 0, 4, 3, 1},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0},
		{2, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 4, 0, 0},
		{0, 0, 0, 0, 3, 0, 0, 0},
		{0, 1, 0, 0, 0, 1, 1, 0},
		{2, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 3, 0, 0, 0},
		{2, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 3, 3, 0},
		{0, 0, 0, 0, 0, 4, 4, 0},
		{1, 0, 0, 0, 0, 4, 4, 0},
		{1, 0, 0, 0, 0, 7, 6, 1},
		{2, 0, 0, 0, 0, 0, 0, 0},
		{2, 0, 0, 0, 0, 4, 4, 0},
		{0, 0, 0, 0, 0, 0, 0, 1},
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 5, 5, 1},
		{1, 0, 0, 0, 0, 5, 5, 1},
		{0, 0, 0, 0, 0, 2, 0, 0},
		{0, 0, 0, 0, 0, 5, 5, 1},
	}

	input = expand(input, map[int]int{
		0: 3,
	})
	perm := rand.Perm(len(input))
	input = permFloat64(input, perm)

	input = normalize(input, map[int]colStats{
		6: {mean: 0, stdev: 0.33},
		7: {mean: 0, stdev: 0.33},
		8: {mean: 0, stdev: 0.33},
	}, nil)
	output := []int{
		0,
		1,
		0,
		1,
		0,
		0,
		0,
		1,
		1,
		1,
		1,
		1,
		1,
		0,
		1,
		1,
		0,
		2,
		1,
		1,
		0,
		1,
		1,
		0,
		1,
		1,
		0,
		1,
		1,
		1,
		0,
		1,
		1,
		1,
		1,
		2,
		0,
		1,
		1,
		1,
		1,
		1,
		1,
		1,
		2,
		2,
		0,
		1,
		2,
		0,
		1,
		1,
		1,
		0,
		0,
		0,
		0,
		1,
		1,
		1,
		2,
		0,
		1,
		2,
		2,
		2,
		2,
		2,
		2,
		2,
		2,
		2,
		1,
		2,
		2,
		2,
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
