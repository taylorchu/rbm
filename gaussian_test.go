package rbm

import (
	"bytes"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestGaussianTrain(t *testing.T) {
	shift1, shift2 := 3.0, -2.0
	m := NewGaussian(2, 11)
	var data [][]float64
	for i := 0; i < 100; i++ {
		data = append(data, []float64{rand.NormFloat64() + shift1, rand.NormFloat64() + shift2})
	}

	m.Train(data, &Option{
		BatchSize: 10,
		Iteration: 20,
		GibbsStep: 10,
	})

	total := 1000
	for _, test := range []struct {
		in []float64
	}{
		{
			in: []float64{shift1, shift2},
		},
		{
			in: []float64{0.1 + shift1, -0.1 + shift2},
		},
		{
			in: []float64{-0.1 + shift1, 0.1 + shift2},
		},
	} {
		var count int
		for i := 0; i < total; i++ {
			got, _ := m.Reconstruct(test.in, 2)
			// mean +- 3 stddev
			if math.Abs(got[0]-shift1) > 3 || math.Abs(got[1]-shift2) > 3 {
				count++
			}
		}
		errRate := float64(count) / float64(total)
		if errRate > 0.02 {
			t.Fatalf("reconstruct error rate %f", errRate)
		}
	}
}

func TestGaussianMarshal(t *testing.T) {
	m := NewGaussian(3, 2)
	buf := new(bytes.Buffer)
	err := m.WriteTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	m2 := NewGaussian(3, 2)
	err = m2.ReadFrom(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m, m2) {
		t.Fatalf("not equal")
	}
}
